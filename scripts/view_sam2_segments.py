#!/usr/bin/env python3
"""Visualize SAM2 segmented regions as semi-transparent colored overlays."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from typing import Any
from urllib import request

import numpy as np
from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PALETTE = [
    (231, 76, 60),
    (46, 204, 113),
    (52, 152, 219),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
    (230, 126, 34),
    (149, 165, 166),
    (243, 156, 18),
    (39, 174, 96),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_path(path_value: str, metadata_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        (metadata_path.parent / path),
        (PROJECT_ROOT / path),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path


def build_overlay(
    base_image: Image.Image,
    region_masks: list[np.ndarray],
    alpha: int,
    hovered_index: int | None = None,
    hover_alpha: int | None = None,
) -> Image.Image:
    base_rgba = base_image.convert("RGBA")
    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))

    for idx, mask_array in enumerate(region_masks):
        color = PALETTE[idx % len(PALETTE)]
        current_alpha = hover_alpha if hovered_index == idx and hover_alpha is not None else alpha
        color_layer = Image.new("RGBA", base_image.size, (*color, current_alpha))
        binary_mask = Image.fromarray((mask_array.astype(np.uint8) * 255), mode="L")
        overlay.paste(color_layer, (0, 0), binary_mask)

    composed = Image.alpha_composite(base_rgba, overlay)
    return composed.convert("RGB")


class SegmentViewer(tk.Tk):
    def __init__(self, metadata_path: Path, overlay_alpha: int, hover_alpha: int):
        super().__init__()
        self.title("SAM2 Segmentation Viewer")
        self.geometry("1600x900")
        self.minsize(1200, 700)

        self.metadata_path = metadata_path
        data = load_json(metadata_path)
        images = data.get("images", []) if isinstance(data, dict) else []
        if not isinstance(images, list) or not images:
            raise RuntimeError(f"No image segmentation entries found in {metadata_path}")

        self.entries = sorted(images, key=lambda entry: str(entry.get("image", "")))
        self.overlay_alpha = max(1, min(255, overlay_alpha))
        self.hover_alpha = max(self.overlay_alpha, min(255, hover_alpha))
        self.current_index = 0
        self.overlay_cache: dict[int, dict[int | None, Image.Image]] = {}
        self.entry_assets_cache: dict[int, dict[str, Any]] = {}
        self.current_hovered_region: int | None = None
        self.rendered_bounds: tuple[int, int, int, int] | None = None
        self.current_photo: ImageTk.PhotoImage | None = None
        self.match_result_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self.searching = False
        self.spinner_index = 0
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.search_thread: threading.Thread | None = None

        self._build_ui()
        self.bind("<Left>", lambda _event: self.show_previous())
        self.bind("<Right>", lambda _event: self.show_next())
        self.bind("<Configure>", lambda _event: self.render_current())
        self.image_label.bind("<Motion>", self.on_mouse_move)
        self.image_label.bind("<Leave>", self.on_mouse_leave)
        self.image_label.bind("<Button-1>", self.on_click)
        self.show_current()

    def _build_ui(self):
        top = tk.Frame(self, padx=12, pady=10)
        top.pack(fill="x")

        self.prev_button = tk.Button(top, text="← Previous", command=self.show_previous, width=12)
        self.prev_button.pack(side="left")

        self.next_button = tk.Button(top, text="Next →", command=self.show_next, width=12)
        self.next_button.pack(side="left", padx=(8, 0))

        self.counter_label = tk.Label(top, text="")
        self.counter_label.pack(side="left", padx=16)

        self.info_label = tk.Label(top, text="", anchor="w", justify="left")
        self.info_label.pack(side="left", padx=12, fill="x", expand=True)

        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.image_container = tk.Frame(self.main_frame, bd=1, relief="solid")
        self.image_container.pack(side="left", fill="both", expand=True)

        self.image_label = tk.Label(self.image_container, compound="center")
        self.image_label.pack(fill="both", expand=True)

        self.results_panel = tk.Frame(self.main_frame, bd=1, relief="solid", width=420)
        self.results_panel.pack(side="right", fill="y", padx=(12, 0))
        self.results_panel.pack_propagate(False)

        self.results_header = tk.Label(self.results_panel, text="Region Matches", font=("TkDefaultFont", 11, "bold"))
        self.results_header.pack(fill="x", padx=10, pady=(10, 8))

        self.results_scroll = tk.Scrollbar(self.results_panel, orient="vertical")
        self.results_scroll.pack(side="right", fill="y")

        self.results_canvas = tk.Canvas(self.results_panel, yscrollcommand=self.results_scroll.set)
        self.results_canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 10))
        self.results_scroll.config(command=self.results_canvas.yview)

        self.results_inner = tk.Frame(self.results_canvas)
        self.results_canvas.create_window((0, 0), window=self.results_inner, anchor="nw")
        self.results_inner.bind("<Configure>", lambda event: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all")))

        self.results_status = tk.Label(self.results_inner, text="Select a region to match it against the texture index.", justify="left", wraplength=300)
        self.results_status.pack(anchor="w", padx=8, pady=8)

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current()

    def show_next(self):
        if self.current_index < len(self.entries) - 1:
            self.current_index += 1
            self.show_current()

    def show_current(self):
        entry = self.entries[self.current_index]
        image_rel = str(entry.get("image", ""))
        regions = entry.get("regions", [])
        if not isinstance(regions, list):
            regions = []

        self.counter_label.config(text=f"Image {self.current_index + 1}/{len(self.entries)}")
        self.info_label.config(
            text=f"{image_rel}\nRegions: {len(regions)}",
        )
        self.current_hovered_region = None
        self.results_status.config(text="Select a region to match it against the texture index.")
        self.clear_match_results()
        self.render_current()

    def _load_entry_assets(self, entry_index: int) -> dict[str, Any]:
        cached = self.entry_assets_cache.get(entry_index)
        if cached is not None:
            return cached

        entry = self.entries[entry_index]
        image_path = resolve_path(str(entry.get("image", "")), self.metadata_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for viewer: {image_path}")
        base_image = Image.open(image_path).convert("RGB")

        regions = entry.get("regions", [])
        if not isinstance(regions, list):
            regions = []

        region_masks: list[np.ndarray] = []
        for region in regions:
            if not isinstance(region, dict):
                continue
            mask_file = region.get("mask_file")
            if isinstance(mask_file, str):
                mask_path = resolve_path(mask_file, self.metadata_path)
                if not mask_path.exists():
                    continue
                mask_image = Image.open(mask_path).convert("L")
                if mask_image.size != base_image.size:
                    mask_image = mask_image.resize(base_image.size, Image.NEAREST)
                mask_array = np.asarray(mask_image) > 0
                mask_image.close()
                region_masks.append(mask_array)

        assets = {"base_image": base_image.copy(), "region_masks": region_masks}
        base_image.close()
        self.entry_assets_cache[entry_index] = assets
        return assets

    def _compose_current(self, hovered_index: int | None) -> Image.Image:
        index_cache = self.overlay_cache.setdefault(self.current_index, {})
        if hovered_index in index_cache:
            return index_cache[hovered_index]

        assets = self._load_entry_assets(self.current_index)
        composed = build_overlay(
            assets["base_image"],
            assets["region_masks"],
            self.overlay_alpha,
            hovered_index=hovered_index,
            hover_alpha=self.hover_alpha,
        )
        index_cache[hovered_index] = composed
        return composed

    def render_current(self):
        try:
            image = self._compose_current(self.current_hovered_region)
        except Exception as exc:
            self.image_label.configure(text=f"Could not render image:\n{exc}", image="")
            self.current_photo = None
            return

        available_w = max(200, self.image_container.winfo_width() - 16)
        available_h = max(200, self.image_container.winfo_height() - 16)
        display = image.copy()
        display.thumbnail((available_w, available_h), Image.Resampling.LANCZOS)
        rendered_w, rendered_h = display.size
        offset_x = max(0, (self.image_container.winfo_width() - rendered_w) // 2)
        offset_y = max(0, (self.image_container.winfo_height() - rendered_h) // 2)
        self.rendered_bounds = (offset_x, offset_y, rendered_w, rendered_h)

        photo = ImageTk.PhotoImage(display)
        self.current_photo = photo
        self.image_label.configure(image=photo, text="")

    def _region_at_display_position(self, x: int, y: int) -> int | None:
        bounds = self.rendered_bounds
        if bounds is None:
            return None
        offset_x, offset_y, rendered_w, rendered_h = bounds
        if x < offset_x or y < offset_y or x >= offset_x + rendered_w or y >= offset_y + rendered_h:
            return None

        assets = self._load_entry_assets(self.current_index)
        base_image = assets["base_image"]
        region_masks = assets["region_masks"]
        if not region_masks:
            return None

        rel_x = x - offset_x
        rel_y = y - offset_y
        img_x = min(base_image.width - 1, max(0, int(rel_x * base_image.width / rendered_w)))
        img_y = min(base_image.height - 1, max(0, int(rel_y * base_image.height / rendered_h)))

        for idx in range(len(region_masks) - 1, -1, -1):
            if bool(region_masks[idx][img_y, img_x]):
                return idx
        return None

    def _start_search_spinner(self):
        if not self.searching:
            return
        spinner = self.spinner_chars[self.spinner_index % len(self.spinner_chars)]
        self.results_status.config(text=f"Searching for matches {spinner}")
        self.spinner_index += 1
        self.after(100, self._start_search_spinner)

    def _stop_search_spinner(self):
        self.searching = False
        self.spinner_index = 0
        self.results_status.config(text="Select a region to match it against the texture index.")

    def clear_match_results(self):
        for child in self.results_inner.winfo_children():
            if child is not self.results_status:
                child.destroy()
        self.results_inner.update_idletasks()

    def download_thumbnail(self, url: str, size: tuple[int, int]) -> Image.Image | None:
        try:
            with request.urlopen(url, timeout=30) as response:
                payload = response.read()
        except Exception:
            return None
        if not payload:
            return None

        downloaded = Image.open(__import__("io").BytesIO(payload)).convert("RGB")
        downloaded.thumbnail(size, Image.Resampling.LANCZOS)
        return downloaded

    def display_match_results(self, payload: dict[str, Any]) -> None:
        self.clear_match_results()
        matches = payload.get("matches", [])
        if not isinstance(matches, list) or not matches:
            self.results_status.config(text="No matching textures found for this region.")
            self.results_status.pack(anchor="w", padx=8, pady=8)
            return

        self.results_status.config(text=f"Top {len(matches)} matches for selected region", justify="left", wraplength=300)
        self.results_status.pack(anchor="w", padx=8, pady=8)

        for idx, match in enumerate(matches, start=1):
            match_frame = tk.Frame(self.results_inner, bd=1, relief="solid", padx=6, pady=6)
            match_frame.pack(fill="x", padx=8, pady=6)

            title = match.get("product_name") or match.get("brand_name") or match.get("product_id") or f"Match {idx}"
            score = match.get("score")
            score_text = f"score={score}" if score is not None else "score=n/a"
            tk.Label(match_frame, text=f"#{idx} {title} ({score_text})", justify="left", anchor="w", wraplength=260, font=("TkDefaultFont", 10, "bold")).pack(fill="x")

            inner = tk.Frame(match_frame)
            inner.pack(fill="x", pady=(4, 0))

            texture_url = match.get("texture_url")
            if texture_url:
                preview = self.download_thumbnail(texture_url, (72, 72))
                if preview is not None:
                    rgba = preview.convert("RGBA")
                    white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    composited = Image.alpha_composite(white_bg, rgba)
                    photo = ImageTk.PhotoImage(composited)
                    label = tk.Label(inner, image=photo)
                    label.image = photo
                    label.pack(side="left", padx=(0, 8), pady=2)

            feature_lines = []
            for key in ["brand_name", "product_category", "material_type", "location", "application", "color"]:
                value = match.get(key)
                if value:
                    value_text = value if isinstance(value, str) else ", ".join(str(item) for item in value)
                    feature_lines.append(f"{key}: {value_text}")

            product_url = match.get("product_url")
            if product_url:
                feature_lines.append(f"product: {product_url}")

            preview_text = "\n".join(feature_lines[:6]) if feature_lines else "No feature metadata"
            details = tk.Label(inner, text=preview_text, justify="left", anchor="w", wraplength=220, font=("TkDefaultFont", 9))
            details.pack(side="left", fill="both", expand=True)

            if product_url:
                details.bind("<Button-1>", lambda _event, url=product_url: webbrowser.open(url))
                details.config(fg="blue", cursor="hand2")

    def on_mouse_move(self, event: tk.Event):
        hovered = self._region_at_display_position(event.x, event.y)
        if hovered != self.current_hovered_region:
            self.current_hovered_region = hovered
            self.render_current()

    def on_mouse_leave(self, _event: tk.Event):
        if self.current_hovered_region is not None:
            self.current_hovered_region = None
            self.render_current()

    def on_click(self, _event: tk.Event):
        if self.current_hovered_region is None:
            return
        self.match_selected_region(self.current_hovered_region)

    def match_selected_region(self, region_index: int) -> None:
        if region_index is None or self.searching:
            return

        entry = self.entries[self.current_index]
        image_path = resolve_path(str(entry.get("image", "")), self.metadata_path)
        if not image_path.exists():
            self.results_status.config(text=f"Image not found: {image_path}")
            return

        assets = self._load_entry_assets(self.current_index)
        region_masks = assets["region_masks"]
        if region_index < 0 or region_index >= len(region_masks):
            self.results_status.config(text="Selected region is out of range.")
            return

        cache_key = (self.current_index, region_index)
        if cache_key in self.match_result_cache:
            self.display_match_results(self.match_result_cache[cache_key])
            return

        self.searching = True
        self.spinner_index = 0
        self.clear_match_results()
        self.results_status.config(text="Searching for matches ⠋")
        self.results_status.pack(anchor="w", padx=8, pady=8)
        self._start_search_spinner()

        mask_array = (region_masks[region_index].astype(np.uint8) * 255)
        mask_image = Image.fromarray(mask_array, mode="L")
        temp_dir = Path(tempfile.gettempdir()) / "lightbeans_match_viewer"
        temp_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        mask_path = temp_dir / f"{stem}_region_{region_index}.png"
        result_path = temp_dir / f"{stem}_region_{region_index}_matches.json"
        mask_image.save(mask_path)

        script_path = PROJECT_ROOT / "scripts" / "match_masked_region_to_texture.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--image",
            str(image_path),
            "--mask",
            str(mask_path),
            "--limit",
            "5",
            "--output",
            str(result_path),
        ]

        def worker() -> None:
            try:
                completed = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                self.after(0, lambda: self._finish_match_search(None, "Matching timed out for this region."))
                return

            if completed.returncode != 0:
                error_text = completed.stderr.strip() or completed.stdout.strip() or "Unknown matching error."
                self.after(0, lambda: self._finish_match_search(None, f"Matching failed:\n{error_text}"))
                return

            if not result_path.exists():
                self.after(0, lambda: self._finish_match_search(None, "Match script did not produce a result file."))
                return

            try:
                payload = load_json(result_path)
            except Exception:
                self.after(0, lambda: self._finish_match_search(None, "Unable to read the match result file."))
                return

            self.match_result_cache[cache_key] = payload
            self.after(0, lambda: self._finish_match_search(payload, None))

        self.search_thread = threading.Thread(target=worker, daemon=True)
        self.search_thread.start()

    def _finish_match_search(self, payload: dict[str, Any] | None, error_message: str | None) -> None:
        if self.search_thread and self.search_thread.is_alive():
            self.search_thread = None
        self.searching = False
        if error_message:
            self.results_status.config(text=error_message)
            self.results_status.pack(anchor="w", padx=8, pady=8)
            return

        if payload is None:
            self.results_status.config(text="Match script did not produce a result file.")
            self.results_status.pack(anchor="w", padx=8, pady=8)
            return

        self.display_match_results(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View SAM2 segmentation overlays.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("results/sam2_segments/segments_metadata.json"),
        help="Path to SAM2 metadata JSON.",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=150,
        help="Overlay alpha (1-255).",
    )
    parser.add_argument(
        "--hover-alpha",
        type=int,
        default=235,
        help="Overlay alpha (1-255) for hovered region.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_path = args.metadata
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    app = SegmentViewer(metadata_path=metadata_path, overlay_alpha=args.alpha, hover_alpha=args.hover_alpha)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
