#!/usr/bin/env python3
"""Visualize SAM2 segmented regions as semi-transparent colored overlays."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from typing import Any
from tkinter import filedialog, messagebox
from urllib import request
from urllib.parse import urljoin, urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from masked_region_texture_matcher import MaskedRegionTextureMatcher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH_PATH = PROJECT_ROOT / "ground_truth.json"
PRODUCT_EXPORT_PATH = PROJECT_ROOT / "data" / "platform-products-production-v2-en.json"
FILTER_VALUES_PATH = PROJECT_ROOT / "data" / "product-field-values-readable.json"
DEFAULT_MATCH_METHOD = "hsi_dinov2"
DEFAULT_MATCH_INDEX = "lightbeans-texture-hsi-dinov2-views-15"
DEFAULT_HSI_PREFILTER_LIMIT = 20
FILTER_CLI_MAP = {
    "location": "--filter-location",
    "application": "--filter-application",
    "productCategory": "--filter-product-category",
    "materialType": "--filter-material-type",
    "color": "--filter-color",
    "brandName": "--filter-brand-name",
}

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


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


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


def build_texture_url(preview_path: str | None) -> str | None:
    if not preview_path:
        return None
    normalized = str(preview_path).strip()
    if not normalized:
        return None
    if normalized.startswith("http://") or normalized.startswith("https://"):
        return normalized
    return urljoin("https://cdn.lightbeans.com/textures/", normalized.lstrip("/"))


def texture_url_key(url: str | None) -> str | None:
    if not url:
        return None
    normalized = str(url).strip()
    if not normalized:
        return None
    parsed = urlparse(normalized)
    path = parsed.path or normalized
    marker = "/texture/"
    marker_index = path.find(marker)
    if marker_index < 0:
        return None
    suffix = path[marker_index + len(marker) :].strip().strip("/")
    if not suffix:
        return None
    return suffix.lower()


def resolve_product_url(record: dict[str, Any] | None) -> str | None:
    if not isinstance(record, dict):
        return None
    direct_url = record.get("link") or record.get("product_url") or record.get("productUrl")
    if direct_url:
        return str(direct_url)
    brand_slug = record.get("brandSlug") or record.get("brand_slug")
    product_slug = record.get("productSlug") or record.get("product_slug")
    if brand_slug and product_slug:
        return f"https://lightbeans.com/en/texture/{brand_slug}/{product_slug}"
    return None


def as_display_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if str(v).strip())
    if isinstance(value, tuple):
        return ", ".join(str(v) for v in value if str(v).strip())
    return str(value)


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


def extract_max_fitting_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    binary = mask.astype(bool)
    if not np.any(binary):
        raise ValueError("Mask is empty; no non-zero pixels were found.")

    height, width = binary.shape
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best_bbox = (0, 0, 1, 1)

    for row_idx in range(height):
        row = binary[row_idx]
        heights = np.where(row, heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for col_idx in range(width + 1):
            current_height = int(heights[col_idx]) if col_idx < width else 0
            start = col_idx
            while stack and stack[-1][1] > current_height:
                left_col, popped_height = stack.pop()
                area = popped_height * (col_idx - left_col)
                if area > best_area:
                    x1 = left_col
                    x2 = col_idx
                    y2 = row_idx + 1
                    y1 = y2 - popped_height
                    best_bbox = (x1, y1, x2, y2)
                    best_area = area
                start = left_col
            if not stack or stack[-1][1] < current_height:
                stack.append((start, current_height))

    return best_bbox


def square_subset_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bbox for square conversion: {bbox}")

    side = min(width, height)
    square_x1 = x1 + (width - side) // 2
    square_y1 = y1 + (height - side) // 2
    square_x2 = square_x1 + side
    square_y2 = square_y1 + side
    return (square_x1, square_y1, square_x2, square_y2)


class SegmentViewer(tk.Tk):
    def __init__(
        self,
        metadata_path: Path,
        overlay_alpha: int,
        hover_alpha: int,
        meili_key: str | None = None,
        use_bbox_for_match: bool = False,
        show_bbox: bool = False,
        log_server_metrics: bool = False,
        selection_mode: str = "rectangle",
    ):
        super().__init__()
        self.title("SAM2 Segmentation Viewer")
        self.geometry("1600x900")
        self.minsize(1200, 700)

        self.metadata_path = metadata_path
        self.meili_key = str(meili_key).strip() if meili_key else None
        self.use_bbox_for_match = bool(use_bbox_for_match)
        self.show_bbox = bool(show_bbox)
        self.log_server_metrics = bool(log_server_metrics)
        self.selection_mode = selection_mode if selection_mode in {"rectangle", "mask"} else "rectangle"
        self.matcher = MaskedRegionTextureMatcher(
            api_key=self.meili_key,
            index_name=DEFAULT_MATCH_INDEX,
            method=DEFAULT_MATCH_METHOD,
            hsi_prefilter_limit=DEFAULT_HSI_PREFILTER_LIMIT,
            limit=10,
            log_server_metrics=self.log_server_metrics,
        )
        data = load_json(metadata_path)
        images = data.get("images", []) if isinstance(data, dict) else []
        if not isinstance(images, list) or not images:
            raise RuntimeError(f"No image segmentation entries found in {metadata_path}")

        self.entries = sorted(images, key=lambda entry: str(entry.get("image", "")))
        self.overlay_alpha = max(1, min(255, overlay_alpha))
        self.hover_alpha = max(self.overlay_alpha, min(255, hover_alpha))
        self.current_index = 0
        self.overlay_cache: dict[int, dict[tuple[int | None, int], Image.Image]] = {}
        self.entry_assets_cache: dict[int, dict[str, Any]] = {}
        self.current_hovered_region: int | None = None
        self.current_pressed_region: int | None = None
        self.pressed_alpha = 255
        self.drag_start_xy: tuple[int, int] | None = None
        self.drag_current_xy: tuple[int, int] | None = None
        self.selected_region_index: int | None = None
        self.selected_crop_bbox: tuple[int, int, int, int] | None = None
        self.rendered_bounds: tuple[int, int, int, int] | None = None
        self.current_photo: ImageTk.PhotoImage | None = None
        self.match_result_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.searching = False
        self.spinner_index = 0
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.search_thread: threading.Thread | None = None
        self.ground_truth_map = self._load_ground_truth_map()
        self.product_catalog = self._load_product_catalog()
        self.filter_entries = self._load_filter_entries()
        self.filter_vars: dict[str, tk.StringVar] = {}
        self.filter_value_to_key: dict[str, dict[str, str]] = {}
        self.goto_index_var = tk.StringVar()
        self._build_ui()
        self.bind("<Left>", lambda _event: self.show_previous())
        self.bind("<Right>", lambda _event: self.show_next())
        self.bind("<Configure>", lambda _event: self.render_current())
        self.image_label.bind("<Motion>", self.on_mouse_move)
        self.image_label.bind("<Leave>", self.on_mouse_leave)
        self.image_label.bind("<ButtonPress-1>", self.on_mouse_down)
        self.image_label.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.show_current()

    def _load_filter_entries(self) -> dict[str, list[tuple[str, str]]]:
        if not FILTER_VALUES_PATH.exists():
            return {}

        try:
            payload = load_json(FILTER_VALUES_PATH)
        except (json.JSONDecodeError, OSError):
            return {}

        if not isinstance(payload, dict):
            return {}

        entries: dict[str, list[tuple[str, str]]] = {}
        for field_name, values in payload.items():
            if not isinstance(values, list):
                continue
            options: list[tuple[str, str]] = []
            for item in values:
                if not isinstance(item, dict):
                    continue
                key = item.get("key")
                value = item.get("value")
                if key is None or value is None:
                    continue
                options.append((str(key), str(value)))
            if options:
                options.sort(key=lambda item: item[1].lower())
                entries[str(field_name)] = [("Any", "Any")] + options
        return entries

    def _build_filter_selectors(self):
        filters_frame = tk.Frame(self, padx=12, pady=0)
        filters_frame.pack(fill="x", pady=(0, 10))

        tk.Label(filters_frame, text="Filters:", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        filter_bar = tk.Frame(filters_frame)
        filter_bar.pack(fill="x", pady=(4, 0))

        for field_name, options in self.filter_entries.items():
            label = field_name.replace("productCategory", "Product Category").replace("materialType", "Material Type").replace("brandName", "Brand Name")
            label = " ".join(part.capitalize() for part in label.replace("_", " ").replace("-", " ").split())
            var = tk.StringVar(value="Any")
            self.filter_vars[field_name] = var
            self.filter_value_to_key[field_name] = {display: key for key, display in options}

            frame = tk.Frame(filter_bar)
            frame.pack(side="left", padx=(0, 10), pady=4)
            tk.Label(frame, text=f"{label}:").pack(anchor="w")
            menu = tk.OptionMenu(frame, var, *[option[1] for option in options])
            menu.config(width=24)
            menu.pack(anchor="w")

    def _selected_filter_key(self, field_name: str) -> str | None:
        var = self.filter_vars.get(field_name)
        if var is None:
            return None
        selected_display = var.get().strip()
        if not selected_display or selected_display == "Any":
            return None
        mapping = self.filter_value_to_key.get(field_name, {})
        selected_key = mapping.get(selected_display)
        if selected_key is None or selected_key == "Any":
            return None
        return selected_key

    def _build_ui(self):
        top = tk.Frame(self, padx=12, pady=10)
        top.pack(fill="x")

        self.prev_button = tk.Button(top, text="← Previous", command=self.show_previous, width=12)
        self.prev_button.pack(side="left")

        self.goto_entry = tk.Entry(top, textvariable=self.goto_index_var, width=6, justify="center")
        self.goto_entry.pack(side="left", padx=(8, 0))
        self.goto_entry.bind("<Return>", self.on_go_to_index)

        self.goto_button = tk.Button(top, text="Go", command=self.go_to_index, width=5)
        self.goto_button.pack(side="left", padx=(6, 0))

        self.next_button = tk.Button(top, text="Next →", command=self.show_next, width=12)
        self.next_button.pack(side="left", padx=(8, 0))

        self.open_image_button = tk.Button(top, text="Open image…", command=self.open_image_from_pc, width=12)
        self.open_image_button.pack(side="left", padx=(8, 0))

        self.counter_label = tk.Label(top, text="")
        self.counter_label.pack(side="left", padx=16)

        self.info_label = tk.Label(top, text="", anchor="w", justify="left")
        self.info_label.pack(side="left", padx=12, fill="x", expand=True)

        if self.filter_entries:
            self._build_filter_selectors()

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

        self.results_inner = tk.Frame(self.results_panel)
        self.results_inner.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.real_match_frame = tk.Frame(self.results_inner, bd=1, relief="solid", padx=6, pady=6)
        self.real_match_frame.pack(fill="x", pady=(0, 8))

        self.real_match_title = tk.Label(self.real_match_frame, text="Real match", justify="left", anchor="w", wraplength=300, font=("TkDefaultFont", 10, "bold"))
        self.real_match_title.pack(fill="x", pady=(0, 4))

        self.real_match_body = tk.Frame(self.real_match_frame, padx=0, pady=0)
        self.real_match_body.pack(fill="x")

        self.match_results_panel = tk.Frame(self.results_inner, bd=1, relief="solid")
        self.match_results_panel.pack(fill="both", expand=True)

        self.results_status = tk.Label(self.match_results_panel, text="Select a region to match it against the texture index.", justify="left", wraplength=300)
        self.results_status.pack(anchor="w", padx=8, pady=(8, 6))

        self.results_scroll = tk.Scrollbar(self.match_results_panel, orient="vertical")
        self.results_scroll.pack(side="right", fill="y")

        self.results_canvas = tk.Canvas(self.match_results_panel, yscrollcommand=self.results_scroll.set)
        self.results_canvas.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 8))
        self.results_scroll.config(command=self.results_canvas.yview)

        self.search_results_frame = tk.Frame(self.results_canvas)
        self.results_canvas.create_window((0, 0), window=self.search_results_frame, anchor="nw")
        self.search_results_frame.bind("<Configure>", lambda event: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all")))

    def _load_ground_truth_map(self) -> dict[str, str]:
        if not GROUND_TRUTH_PATH.exists():
            return {}
        payload = load_json(GROUND_TRUTH_PATH)
        if isinstance(payload, dict):
            return {str(filename): str(product_id) for filename, product_id in payload.items()}
        if not isinstance(payload, list):
            return {}

        catalog: dict[str, str] = {}
        for record in payload:
            if not isinstance(record, dict):
                continue
            product_id = record.get("id")
            if product_id is None:
                continue
            image_name = record.get("image_name") or record.get("imageName")
            if isinstance(image_name, str) and image_name.strip():
                catalog[image_name.strip()] = str(product_id)
        return catalog

    def _load_product_catalog(self) -> dict[str, dict[str, Any]]:
        if not PRODUCT_EXPORT_PATH.exists():
            return {}
        payload = load_json(PRODUCT_EXPORT_PATH)
        if not isinstance(payload, list):
            return {}
        catalog: dict[str, dict[str, Any]] = {}
        for record in payload:
            if not isinstance(record, dict):
                continue
            product_id = record.get("id")
            if product_id is None:
                continue
            catalog[str(product_id)] = record
        return catalog

    def _resolve_current_ground_truth_product(self) -> dict[str, Any] | None:
        entry = self.entries[self.current_index]
        image_rel = str(entry.get("image", ""))
        candidates = []
        if image_rel:
            candidates.extend([
                Path(image_rel).name,
                image_rel,
                str(Path(image_rel).as_posix()),
            ])
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                product_id = self.ground_truth_map.get(candidate)
                if product_id:
                    return self.product_catalog.get(str(product_id))

        alt_candidates = []
        if image_rel:
            file_name = Path(image_rel).name
            alt_candidates.extend([
                file_name,
                file_name.replace(".png", ""),
                file_name.replace(".jpg", ""),
                file_name.replace(".jpeg", ""),
                file_name.replace(".webp", ""),
            ])
        for candidate in alt_candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                product_id = self.ground_truth_map.get(candidate)
                if product_id:
                    return self.product_catalog.get(str(product_id))

        fallback_keys = list(self.product_catalog.keys())
        for key in fallback_keys:
            record = self.product_catalog.get(key)
            if not isinstance(record, dict):
                continue
            image_name_value = record.get("image_name") or record.get("imageName")
            if image_name_value and Path(image_rel).name == str(image_name_value):
                return record

        return None

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

        self.selected_region_index = None
        self.selected_crop_bbox = None
        self.drag_start_xy = None
        self.drag_current_xy = None
        self.goto_index_var.set(str(self.current_index + 1))
        self.counter_label.config(text=f"Image {self.current_index + 1}/{len(self.entries)}")
        self.info_label.config(
            text=f"{image_rel}\nRegions: {len(regions)}",
        )
        self.current_hovered_region = None
        self.results_status.config(text=self._selection_instruction_text())
        self.render_real_match()
        self.clear_match_results()
        self.render_current()

    def on_go_to_index(self, _event: tk.Event) -> None:
        self.go_to_index()

    def go_to_index(self) -> None:
        raw = self.goto_index_var.get().strip()
        if not raw:
            return
        if not raw.isdigit():
            self.results_status.config(text=f"Invalid page '{raw}'. Enter a number between 1 and {len(self.entries)}.")
            return

        target = int(raw)
        if target < 1 or target > len(self.entries):
            self.results_status.config(text=f"Page out of range. Enter a number between 1 and {len(self.entries)}.")
            return

        self.current_index = target - 1
        self.show_current()

    def open_image_from_pc(self) -> None:
        selected_path = filedialog.askopenfilename(
            title="Open image for analysis",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not selected_path:
            return
        image_path = Path(selected_path)
        if not image_path.exists():
            messagebox.showerror("Image not found", f"Selected image does not exist:\n{image_path}")
            return
        try:
            test_image = Image.open(image_path).convert("RGB")
            test_image.close()
        except Exception as exc:
            messagebox.showerror("Invalid image", f"Could not open selected image:\n{exc}")
            return

        self.entries.append({"image": str(image_path), "regions": []})
        self.current_index = len(self.entries) - 1
        self.overlay_cache.pop(self.current_index, None)
        self.entry_assets_cache.pop(self.current_index, None)
        self.show_current()

    def render_real_match(self) -> None:
        for child in self.real_match_body.winfo_children():
            child.destroy()

        product = self._resolve_current_ground_truth_product()
        if product is None:
            tk.Label(self.real_match_body, text="No ground-truth match found for this image.", justify="left", anchor="w", wraplength=280).pack(fill="x")
            return

        product_name = product.get("productName") or product.get("name") or "Unknown product"
        brand_name = product.get("brandName") or product.get("brand_name") or "Unknown brand"
        product_url = resolve_product_url(product)

        texture_url = None
        preview_path = product.get("previewPath") or product.get("image_url") or product.get("imageUrl")
        if preview_path:
            if isinstance(preview_path, str) and (preview_path.startswith("http://") or preview_path.startswith("https://")):
                texture_url = preview_path
            else:
                texture_url = build_texture_url(str(preview_path))

        row = tk.Frame(self.real_match_body)
        row.pack(fill="x")

        if texture_url:
            preview = self.download_thumbnail(texture_url, (72, 72))
            if preview is not None:
                rgba = preview.convert("RGBA")
                white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                composited = Image.alpha_composite(white_bg, rgba)
                photo = ImageTk.PhotoImage(composited)
                image_label = tk.Label(row, image=photo)
                image_label.image = photo
                image_label.pack(side="left", padx=(0, 8), pady=2)

        content = tk.Frame(row)
        content.pack(side="left", fill="both", expand=True)

        title = tk.Label(content, text=f"{brand_name} — {product_name}", justify="left", anchor="w", wraplength=200, font=("TkDefaultFont", 10, "bold"))
        title.pack(fill="x")
        if product_url:
            title.bind("<Button-1>", lambda _event, url=product_url: webbrowser.open(url))
            title.config(fg="blue", cursor="hand2")

        details = []
        if product.get("productCategory"):
            details.append(f"Category: {', '.join(str(v) for v in product.get('productCategory', []))}")
        if product.get("product_category"):
            details.append(f"Category: {as_display_list(product.get('product_category'))}")
        if product.get("materialType"):
            details.append(f"Material: {', '.join(str(v) for v in product.get('materialType', []))}")
        if product.get("material_type"):
            details.append(f"Material: {as_display_list(product.get('material_type'))}")
        if product.get("color"):
            details.append(f"Color: {', '.join(str(v) for v in product.get('color', []))}")
        if product.get("image_name"):
            details.append(f"Image: {product.get('image_name')}")
        if details:
            info = tk.Label(content, text="\n".join(details), justify="left", anchor="w", wraplength=220, font=("TkDefaultFont", 9))
            info.pack(fill="x", pady=(4, 0))

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

    def _compose_current(self, hovered_index: int | None, active_alpha: int) -> Image.Image:
        index_cache = self.overlay_cache.setdefault(self.current_index, {})
        cache_key = (hovered_index, active_alpha)
        if cache_key in index_cache:
            return index_cache[cache_key]

        assets = self._load_entry_assets(self.current_index)
        composed = build_overlay(
            assets["base_image"],
            assets["region_masks"],
            self.overlay_alpha,
            hovered_index=hovered_index,
            hover_alpha=active_alpha,
        )
        index_cache[cache_key] = composed
        return composed

    def _selected_region_bbox(self) -> tuple[int, int, int, int] | None:
        if self.selected_crop_bbox is not None:
            return self.selected_crop_bbox
        if self.selected_region_index is None:
            return None
        assets = self._load_entry_assets(self.current_index)
        region_masks = assets["region_masks"]
        if self.selected_region_index < 0 or self.selected_region_index >= len(region_masks):
            return None
        mask = region_masks[self.selected_region_index]
        try:
            return square_subset_from_bbox(extract_max_fitting_bbox(mask))
        except ValueError:
            return None

    def render_current(self):
        if self.selection_mode == "mask":
            active_region = self.current_pressed_region if self.current_pressed_region is not None else self.current_hovered_region
            active_alpha = self.pressed_alpha if self.current_pressed_region is not None else self.hover_alpha
        else:
            active_region = None
            active_alpha = self.hover_alpha
        try:
            image = self._compose_current(active_region, active_alpha)
        except Exception as exc:
            self.image_label.configure(text=f"Could not render image:\n{exc}", image="")
            self.current_photo = None
            return

        available_w = max(200, self.image_container.winfo_width() - 16)
        available_h = max(200, self.image_container.winfo_height() - 16)
        display = image.copy()
        display.thumbnail((available_w, available_h), Image.Resampling.LANCZOS)
        if self.show_bbox or self.selection_mode == "rectangle":
            bbox = self._selected_region_bbox()
            if bbox is not None:
                self._draw_display_bbox(display, image, bbox, colors=((255, 255, 255), (255, 69, 58)))
        if self.selection_mode == "rectangle" and self.drag_start_xy and self.drag_current_xy:
            x1 = min(self.drag_start_xy[0], self.drag_current_xy[0])
            y1 = min(self.drag_start_xy[1], self.drag_current_xy[1])
            x2 = max(self.drag_start_xy[0], self.drag_current_xy[0]) + 1
            y2 = max(self.drag_start_xy[1], self.drag_current_xy[1]) + 1
            self._draw_display_bbox(display, image, (x1, y1, x2, y2), colors=((255, 255, 255), (255, 215, 0)))
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

    def _display_to_image_position(self, x: int, y: int) -> tuple[int, int] | None:
        bounds = self.rendered_bounds
        if bounds is None:
            return None
        offset_x, offset_y, rendered_w, rendered_h = bounds
        if x < offset_x or y < offset_y or x >= offset_x + rendered_w or y >= offset_y + rendered_h:
            return None
        assets = self._load_entry_assets(self.current_index)
        base_image = assets["base_image"]
        rel_x = x - offset_x
        rel_y = y - offset_y
        img_x = min(base_image.width - 1, max(0, int(rel_x * base_image.width / rendered_w)))
        img_y = min(base_image.height - 1, max(0, int(rel_y * base_image.height / rendered_h)))
        return (img_x, img_y)

    def _draw_display_bbox(
        self,
        display: Image.Image,
        image: Image.Image,
        bbox: tuple[int, int, int, int],
        *,
        colors: tuple[tuple[int, int, int], tuple[int, int, int]],
    ) -> None:
        x1, y1, x2, y2 = bbox
        scale_x = display.width / max(1, image.width)
        scale_y = display.height / max(1, image.height)
        dx1 = max(0, min(display.width - 1, int(round(x1 * scale_x))))
        dy1 = max(0, min(display.height - 1, int(round(y1 * scale_y))))
        dx2 = max(dx1, min(display.width - 1, int(round(x2 * scale_x)) - 1))
        dy2 = max(dy1, min(display.height - 1, int(round(y2 * scale_y)) - 1))
        drawer = ImageDraw.Draw(display)
        drawer.rectangle([(dx1, dy1), (dx2, dy2)], outline=colors[0], width=3)
        drawer.rectangle([(dx1, dy1), (dx2, dy2)], outline=colors[1], width=2)

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
        self.results_status.config(text=self._selection_instruction_text())

    def _selection_instruction_text(self) -> str:
        if self.selection_mode == "rectangle":
            return "Drag to draw a rectangle on the image to match that region."
        return "Click a segmented mask region to match it against the texture index."

    def clear_match_results(self):
        for child in self.search_results_frame.winfo_children():
            child.destroy()
        self.search_results_frame.update_idletasks()

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
        total_time_text = None
        timing = payload.get("timing") if isinstance(payload, dict) else None
        if isinstance(timing, dict):
            try:
                total_time_text = f"Search total: {float(timing.get('total_seconds', 0.0)):.3f}s"
            except (TypeError, ValueError):
                total_time_text = None
        matches = payload.get("matches", [])
        if not isinstance(matches, list) or not matches:
            status_text = "No matching textures found for this region."
            if total_time_text:
                status_text = f"{total_time_text}\n{status_text}"
            self.results_status.config(text=status_text, justify="left", wraplength=300)
            self.results_status.pack(anchor="w", padx=8, pady=(8, 6))
            return

        status_lines = []
        if total_time_text:
            status_lines.append(total_time_text)
        status_lines.append(f"Top {len(matches)} matches for selected region")
        self.results_status.config(text="\n".join(status_lines), justify="left", wraplength=300)
        self.results_status.pack(anchor="w", padx=8, pady=(8, 6))

        real_product = self._resolve_current_ground_truth_product()
        real_texture_key = None
        if isinstance(real_product, dict):
            real_texture_key = texture_url_key(resolve_product_url(real_product))

        for idx, match in enumerate(matches, start=1):
            product_url = resolve_product_url(match)
            match_texture_key = texture_url_key(product_url)
            is_real_match = bool(real_texture_key and match_texture_key and real_texture_key == match_texture_key)

            border_width = 5 if is_real_match else 1
            border_color = "#22c55e" if is_real_match else "#808080"
            match_frame = tk.Frame(
                self.search_results_frame,
                bd=border_width,
                relief="solid",
                padx=6,
                pady=6,
                highlightthickness=border_width,
                highlightbackground=border_color,
                highlightcolor=border_color,
            )
            match_frame.pack(fill="x", padx=8, pady=6)

            product_name = match.get("name") or match.get("product_name") or match.get("product_id") or f"Match {idx}"
            brand_name = match.get("brandName") or match.get("brand_name") or "Unknown brand"
            product_id = match.get("product_id") or match.get("id")
            image_url = match.get("image_url") or match.get("texture_url") or match.get("preview_path")
            if image_url and not (isinstance(image_url, str) and (image_url.startswith("http://") or image_url.startswith("https://"))):
                image_url = build_texture_url(str(image_url))

            score = match.get("score")
            if score is not None:
                try:
                    score_value = float(score)
                    score_label = f"score={score_value:.4f}"
                except (TypeError, ValueError):
                    score_label = f"score={score}"
            else:
                score_label = "score=n/a"

            inner = tk.Frame(match_frame)
            inner.pack(fill="x", pady=(4, 0))

            left = tk.Frame(inner)
            left.pack(side="left", padx=(0, 8), pady=2)
            if image_url:
                preview = self.download_thumbnail(image_url, (72, 72))
                if preview is not None:
                    rgba = preview.convert("RGBA")
                    white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    composited = Image.alpha_composite(white_bg, rgba)
                    photo = ImageTk.PhotoImage(composited)
                    image_label = tk.Label(left, image=photo, bd=0)
                    image_label.image = photo
                    image_label.pack()
                    if product_url:
                        image_label.bind("<Button-1>", lambda _event, url=product_url: webbrowser.open(url))
                        image_label.config(cursor="hand2")

            right = tk.Frame(inner)
            right.pack(side="left", fill="both", expand=True)

            if product_url:
                name_label = tk.Label(
                    right,
                    text=f"#{idx} {brand_name} — {product_name} ({score_label})",
                    justify="left",
                    anchor="w",
                    wraplength=220,
                    font=("TkDefaultFont", 10, "bold"),
                    fg="blue",
                    cursor="hand2",
                )
                name_label.pack(fill="x")
                name_label.bind("<Button-1>", lambda _event, url=product_url: webbrowser.open(url))
            else:
                tk.Label(
                    right,
                    text=f"#{idx} {brand_name} — {product_name} ({score_label})",
                    justify="left",
                    anchor="w",
                    wraplength=220,
                    font=("TkDefaultFont", 10, "bold"),
                ).pack(fill="x")

            detail_lines: list[str] = []
            # if product_id:
            #     detail_lines.append(f"Product ID: {product_id}")
            if detail_lines:
                tk.Label(
                    right,
                    text="\n".join(detail_lines),
                    justify="left",
                    anchor="w",
                    wraplength=220,
                    font=("TkDefaultFont", 9),
                ).pack(fill="x", pady=(2, 0))

    def on_mouse_move(self, event: tk.Event):
        if self.selection_mode == "rectangle":
            if self.drag_start_xy is not None:
                current_pos = self._display_to_image_position(event.x, event.y)
                if current_pos is not None and current_pos != self.drag_current_xy:
                    self.drag_current_xy = current_pos
                    self.render_current()
            return
        hovered = self._region_at_display_position(event.x, event.y)
        if hovered != self.current_hovered_region:
            self.current_hovered_region = hovered
            self.render_current()

    def on_mouse_leave(self, _event: tk.Event):
        if self.selection_mode == "rectangle":
            return
        if self.current_pressed_region is None and self.current_hovered_region is not None:
            self.current_hovered_region = None
            self.render_current()

    def on_mouse_down(self, event: tk.Event):
        if self.selection_mode == "rectangle":
            start_pos = self._display_to_image_position(event.x, event.y)
            self.drag_start_xy = start_pos
            self.drag_current_xy = start_pos
            self.render_current()
            return
        pressed = self._region_at_display_position(event.x, event.y)
        self.current_pressed_region = pressed
        if pressed is not None:
            self.current_hovered_region = pressed
        self.render_current()

    def on_mouse_up(self, event: tk.Event):
        if self.selection_mode == "rectangle":
            if self.drag_start_xy is None:
                return
            end_pos = self._display_to_image_position(event.x, event.y)
            start_pos = self.drag_start_xy
            self.drag_start_xy = None
            self.drag_current_xy = None
            if end_pos is None:
                self.render_current()
                return
            x1 = min(start_pos[0], end_pos[0])
            y1 = min(start_pos[1], end_pos[1])
            x2 = max(start_pos[0], end_pos[0]) + 1
            y2 = max(start_pos[1], end_pos[1]) + 1
            if (x2 - x1) < 3 or (y2 - y1) < 3:
                self.results_status.config(text="Rectangle too small. Drag a larger region.")
                self.results_status.pack(anchor="w", padx=8, pady=(8, 6))
                self.render_current()
                return
            selected_bbox = (x1, y1, x2, y2)
            self.selected_crop_bbox = selected_bbox
            self.render_current()
            self.match_selected_bbox(selected_bbox)
            return
        released = self._region_at_display_position(event.x, event.y)
        pressed = self.current_pressed_region
        self.current_pressed_region = None
        self.render_current()
        if pressed is None or released != pressed:
            return
        self.selected_region_index = pressed
        self.render_current()
        self.match_selected_region(pressed)

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

        try:
            max_bbox = extract_max_fitting_bbox(region_masks[region_index])
            selected_bbox = square_subset_from_bbox(max_bbox)
        except ValueError:
            self.results_status.config(text="Selected region is empty.")
            return
        self.selected_crop_bbox = selected_bbox
        self.render_current()

        selected_filters, filter_signature = self._collect_selected_filters()

        cache_key = (self.current_index, region_index, filter_signature, self.use_bbox_for_match)
        if cache_key in self.match_result_cache:
            cached_payload = self.match_result_cache[cache_key]
            self._apply_bbox_from_payload(cached_payload)
            self.render_current()
            self.display_match_results(cached_payload)
            return

        mask_array = (region_masks[region_index].astype(np.uint8) * 255)
        base_image_array = np.asarray(assets["base_image"], dtype=np.uint8)
        provided_bbox = selected_bbox if self.use_bbox_for_match else None
        filter_values = {
            "location": selected_filters.get("location"),
            "application": selected_filters.get("application"),
            "product_category": selected_filters.get("productCategory"),
            "material_type": selected_filters.get("materialType"),
            "color": selected_filters.get("color"),
            "brand_name": selected_filters.get("brandName"),
        }
        self._start_match_search(
            cache_key=cache_key,
            image_label=str(image_path),
            mask_label=f"{image_path.name}:region-{region_index}",
            image_array=base_image_array,
            mask_array=mask_array,
            provided_bbox=provided_bbox,
            filter_values=filter_values,
        )

    def match_selected_bbox(self, selected_bbox: tuple[int, int, int, int]) -> None:
        if self.searching:
            return
        entry = self.entries[self.current_index]
        image_path = resolve_path(str(entry.get("image", "")), self.metadata_path)
        if not image_path.exists():
            self.results_status.config(text=f"Image not found: {image_path}")
            return
        assets = self._load_entry_assets(self.current_index)
        base_image = assets["base_image"]
        base_image_array = np.asarray(base_image, dtype=np.uint8)
        full_mask_array = np.full((base_image.height, base_image.width), 255, dtype=np.uint8)

        selected_filters, filter_signature = self._collect_selected_filters()
        cache_key = (self.current_index, "bbox", selected_bbox, filter_signature)
        if cache_key in self.match_result_cache:
            cached_payload = self.match_result_cache[cache_key]
            self._apply_bbox_from_payload(cached_payload)
            self.render_current()
            self.display_match_results(cached_payload)
            return

        filter_values = {
            "location": selected_filters.get("location"),
            "application": selected_filters.get("application"),
            "product_category": selected_filters.get("productCategory"),
            "material_type": selected_filters.get("materialType"),
            "color": selected_filters.get("color"),
            "brand_name": selected_filters.get("brandName"),
        }
        self._start_match_search(
            cache_key=cache_key,
            image_label=str(image_path),
            mask_label=f"{image_path.name}:bbox",
            image_array=base_image_array,
            mask_array=full_mask_array,
            provided_bbox=selected_bbox,
            filter_values=filter_values,
        )

    def _collect_selected_filters(self) -> tuple[dict[str, str | None], tuple[tuple[str, str | None], ...]]:
        selected_filters: dict[str, str | None] = {}
        for field_name in FILTER_CLI_MAP:
            selected_filters[field_name] = self._selected_filter_key(field_name)
        filter_signature = tuple((field_name, selected_filters[field_name]) for field_name in FILTER_CLI_MAP)
        return selected_filters, filter_signature

    def _log_timing_to_terminal(self, payload: dict[str, Any]) -> None:
        timing = payload.get("timing") if isinstance(payload, dict) else None
        if not isinstance(timing, dict):
            return
        timing_parts = []
        for key, label in (
            ("hsi_embedding_seconds", "HSI emb"),
            ("hsi_search_seconds", "HSI search"),
            ("dino_query_seconds", "DINO query"),
            ("rerank_seconds", "rerank"),
            ("total_seconds", "total"),
        ):
            try:
                value = float(timing.get(key, 0.0))
            except (TypeError, ValueError):
                continue
            timing_parts.append(f"{label} {value:.3f}s")
        if timing_parts:
            sys.stdout.write("Timing:\n")
            for part in timing_parts:
                sys.stdout.write(f"  {part}\n")
            sys.stdout.flush()

    def _start_match_search(
        self,
        *,
        cache_key: tuple[Any, ...],
        image_label: str,
        mask_label: str,
        image_array: np.ndarray,
        mask_array: np.ndarray,
        provided_bbox: tuple[int, int, int, int] | None,
        filter_values: dict[str, str | None],
    ) -> None:
        self.searching = True
        self.spinner_index = 0
        self.clear_match_results()
        self.results_status.config(text="Searching for matches (HSI -> DINOv2) ⠋")
        self.results_status.pack(anchor="w", padx=8, pady=(8, 6))
        self._start_search_spinner()

        def worker() -> None:
            try:
                payload = self.matcher.match_from_arrays(
                    image_array=image_array,
                    mask_array=mask_array,
                    image_label=image_label,
                    mask_label=mask_label,
                    margin=0,
                    provided_bbox=provided_bbox,
                    filter_values=filter_values,
                )
            except Exception as exc:
                self.after(0, lambda: self._finish_match_search(None, f"Matching failed:\n{exc}"))
                return
            self._log_timing_to_terminal(payload)
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
            self.results_status.pack(anchor="w", padx=8, pady=(8, 6))
            return

        if payload is None:
            self.results_status.config(text="Matcher did not return a result.")
            self.results_status.pack(anchor="w", padx=8, pady=(8, 6))
            return

        self._apply_bbox_from_payload(payload)
        self.render_current()
        self.display_match_results(payload)

    def _apply_bbox_from_payload(self, payload: dict[str, Any]) -> None:
        bbox = payload.get("bbox") if isinstance(payload, dict) else None
        if not isinstance(bbox, dict):
            self.selected_crop_bbox = None
            return
        try:
            x1 = int(bbox.get("x1"))
            y1 = int(bbox.get("y1"))
            x2 = int(bbox.get("x2"))
            y2 = int(bbox.get("y2"))
        except (TypeError, ValueError):
            self.selected_crop_bbox = None
            return
        if x2 <= x1 or y2 <= y1:
            self.selected_crop_bbox = None
            return
        self.selected_crop_bbox = (x1, y1, x2, y2)


def parse_args() -> argparse.Namespace:
    env_values = load_env(PROJECT_ROOT / ".env")
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
        default=0,
        help="Overlay alpha (1-255).",
    )
    parser.add_argument(
        "--hover-alpha",
        type=int,
        default=125,
        help="Overlay alpha (1-255) for hovered region.",
    )
    parser.add_argument(
        "--key",
        default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"),
        help="Meilisearch API key passed to matcher.",
    )
    parser.add_argument(
        "--use-bbox-for-match",
        action="store_true",
        help="Send bbox coordinates to matcher (default: send only selected mask area).",
    )
    parser.add_argument(
        "--show-bbox",
        action="store_true",
        help="Show selected bbox overlay in the viewer.",
    )
    parser.add_argument(
        "--log-server-metrics",
        action="store_true",
        help="Log Meilisearch request/response sizes and latency for each matcher HTTP call.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["rectangle", "mask"],
        default="rectangle",
        help="Region selection mode: rectangle drag (default) or SAM2 mask click.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_path = args.metadata
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    app = SegmentViewer(
        metadata_path=metadata_path,
        overlay_alpha=args.alpha,
        hover_alpha=args.hover_alpha,
        meili_key=args.key,
        use_bbox_for_match=args.use_bbox_for_match,
        show_bbox=args.show_bbox,
        log_server_metrics=args.log_server_metrics,
        selection_mode=args.selection_mode,
    )
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
