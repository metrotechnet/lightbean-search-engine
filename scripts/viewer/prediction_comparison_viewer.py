#!/usr/bin/env python3
"""Interactive viewer to compare SigLIP2 and DINOv2 predictions against the real product record and Meilisearch matches."""

from __future__ import annotations

import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any
from urllib import request, error

from PIL import Image, ImageTk


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


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


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def build_query_from_prediction(prediction: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ["location", "application", "productCategory", "materialType", "color"]:
        best = prediction.get(key, {}).get("best") if isinstance(prediction.get(key), dict) else None
        if best:
            values.append(str(best))
    return " ".join(values)


def prepare_product_dict(product_records: list[dict[str, Any]] | dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    product_map: dict[str, dict[str, Any]] = {}
    if isinstance(product_records, dict):
        items = product_records.values()
    else:
        items = product_records

    for rec in items:
        if not isinstance(rec, dict):
            continue
        product_id = rec.get("id")
        if product_id is not None:
            product_map[str(product_id)] = rec
    return product_map


def fetch_meilisearch(host: str, api_key: str | None, index: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "q": query,
        "limit": limit,
        "attributesToRetrieve": ["product_id", "location", "application", "product_category", "material_type", "color"],
    }
    endpoint = f"{host.rstrip('/')}/indexes/{index}/search"
    req = request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body) if body else {}
            return list(data.get("hits", []))
    except (error.URLError, error.HTTPError) as exc:
        return [{"error": str(exc), "product_id": None}]


def product_field_summary(record: dict[str, Any], key: str) -> str:
    values = as_list(record.get(key, []))
    return ", ".join(values) if values else "-"


def attribute_keys() -> list[str]:
    return ["location", "application", "productCategory", "materialType", "color"]


def build_label_text(prediction: dict[str, Any]) -> str:
    if not prediction:
        return "No prediction"
    lines: list[str] = []
    for key in attribute_keys():
        block = prediction.get(key, {}) if isinstance(prediction.get(key), dict) else {}
        if not block:
            continue
        best = block.get("best")
        top = block.get("top", [])
        if best is not None:
            top_text = ", ".join(f"{item['label']} ({item['score']:.4f})" for item in top[:3])
            lines.append(f"{key}: {best}\n  top: {top_text}")
    return "\n".join(lines) if lines else "No prediction values"


def build_attribute_rows_for_record(record: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key in attribute_keys():
        rows.append((key, product_field_summary(record, key)))
    return rows


def build_prediction_rows(prediction: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key in attribute_keys():
        block = prediction.get(key, {}) if isinstance(prediction.get(key), dict) else {}
        best = block.get("best")
        if best is None:
            rows.append((key, "-"))
            continue
        top = block.get("top", [])
        top_values = "; ".join(
            f"{item.get('label', '-')}={float(item.get('score', 0.0)):.4f}"
            for item in top[:3]
        ) if top else "-"
        rows.append((key, f"{best} | {top_values}"))
    return rows


def build_search_hit_rows(hits: list[dict[str, Any]]) -> list[tuple[str, str, str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for index, hit in enumerate(hits[:5], start=1):
        if "error" in hit:
            rows.append((str(index), hit.get("error", "-"), "-", "-", "-", "-", "-"))
            continue
        rows.append(
            (
                str(index),
                str(hit.get("product_id") or hit.get("id") or "unknown"),
                ", ".join(as_list(hit.get("location"))) or "-",
                ", ".join(as_list(hit.get("application"))) or "-",
                ", ".join(as_list(hit.get("product_category"))) or "-",
                ", ".join(as_list(hit.get("material_type"))) or "-",
                ", ".join(as_list(hit.get("color"))) or "-",
            )
        )
    return rows


class PredictionViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lightbeans Product Attribute Viewer")
        self.geometry("1920x1020")
        self.minsize(1680, 860)

        style = __import__("tkinter.ttk", fromlist=["Style"]).Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        bg = "#f3f6fb"
        panel = "#ffffff"
        accent = "#2557d6"
        accent_soft = "#eaf1ff"
        text = "#1d2433"
        muted = "#5d6982"
        border = "#d8e1f0"

        self.configure(bg=bg)
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=text, font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground=text)
        style.configure("Muted.TLabel", foreground=muted)
        style.configure("Accent.TButton", background=accent, foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#1d49b5")])

        env_values = load_env(PROJECT_ROOT / ".env")
        self.host = env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700"
        self.api_key = env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY")
        self.index = env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or "lightbean-index"

        self.images_dir = PROJECT_ROOT / "images"
        self.siglip_predictions = load_json(PROJECT_ROOT / "SigLIP2_image_predictions.json")
        self.dinov2_predictions = load_json(PROJECT_ROOT / "DINOv2_image_predictions.json")
        self.ground_truth = load_json(PROJECT_ROOT / "ground_truth.json")
        self.product_export = load_json(PROJECT_ROOT / "data" / "platform-products-production-v2-en.json")
        self.product_lookup = prepare_product_dict(self.product_export)

        self.image_names = sorted(
            [p.name for p in self.images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
        )
        self.current_image = tk.StringVar(value=self.image_names[0] if self.image_names else "")

        self.bind("<Left>", lambda _event: self.show_previous_image())
        self.bind("<Right>", lambda _event: self.show_next_image())

        top = tk.Frame(self, bg=bg, padx=14, pady=12)
        top.pack(fill="x")

        controls = tk.Frame(top, bg=panel, bd=1, relief="solid")
        controls.pack(fill="x", padx=0, pady=0)

        self.prev_button = tk.Button(controls, text="← Previous", command=self.show_previous_image, width=14, bg=accent_soft, fg=accent, activebackground="#d7e8ff", relief="flat", font=("Segoe UI", 10, "bold"), bd=0)
        self.prev_button.pack(side="left", padx=(12, 8), pady=10)

        self.image_selector = ttk = __import__("tkinter.ttk", fromlist=["Combobox"]).Combobox(controls, values=self.image_names, textvariable=self.current_image, state="readonly", width=56)
        self.image_selector.pack(side="left", padx=4, pady=10)
        self.image_selector.bind("<<ComboboxSelected>>", self.on_image_change)

        self.next_button = tk.Button(controls, text="Next →", command=self.show_next_image, width=14, bg=accent, fg="white", activebackground="#1d49b5", relief="flat", font=("Segoe UI", 10, "bold"), bd=0)
        self.next_button.pack(side="left", padx=(8, 12), pady=10)

        self.counter_label = tk.Label(top, text="", bg=bg, fg=muted, font=("Segoe UI", 10, "bold"))
        self.counter_label.pack(anchor="w", padx=14, pady=(0, 10))

        main = tk.Frame(self, bg=bg)
        main.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        main.grid_columnconfigure(0, weight=0, minsize=290)
        main.grid_columnconfigure(1, weight=1, minsize=380, uniform="top-panels")
        main.grid_columnconfigure(2, weight=1, minsize=380, uniform="top-panels")
        main.grid_columnconfigure(3, weight=1, minsize=380, uniform="top-panels")
        main.grid_rowconfigure(0, weight=1, uniform="main-rows")
        main.grid_rowconfigure(1, weight=1, uniform="main-rows")

        preview_outer = tk.Frame(main, bg=panel, bd=1, relief="solid")
        preview_outer.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10), pady=4)
        preview_outer.grid_rowconfigure(0, weight=1)
        preview_outer.grid_columnconfigure(0, weight=1)
        self.preview_canvas = tk.Label(preview_outer, bg=panel, compound="center", padx=4, pady=4)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")

        self.actual_frame = tk.Frame(main, bg=panel, bd=1, relief="solid")
        self.actual_frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self.actual_frame.grid_propagate(True)
        self.actual_frame.grid_rowconfigure(1, weight=1)
        self.actual_frame.grid_columnconfigure(0, weight=1)
        self.actual_header = tk.Label(self.actual_frame, text="Actual product", bg=panel, fg=text, font=("Segoe UI", 11, "bold"), anchor="w")
        self.actual_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        self.actual_table = self._build_rows_panel(self.actual_frame, columns=("Attribute", "Value"))

        self.siglip_frame = tk.Frame(main, bg=panel, bd=1, relief="solid")
        self.siglip_frame.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)
        self.siglip_frame.grid_propagate(True)
        self.siglip_frame.grid_rowconfigure(1, weight=1)
        self.siglip_frame.grid_columnconfigure(0, weight=1)
        self.siglip_header = tk.Label(self.siglip_frame, text="SigLIP2", bg=panel, fg=text, font=("Segoe UI", 11, "bold"), anchor="w")
        self.siglip_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        self.siglip_table = self._build_rows_panel(self.siglip_frame, columns=("Attribute", "Best / Top3"))

        self.dinov2_frame = tk.Frame(main, bg=panel, bd=1, relief="solid")
        self.dinov2_frame.grid(row=0, column=3, sticky="nsew", padx=4, pady=4)
        self.dinov2_frame.grid_propagate(True)
        self.dinov2_frame.grid_rowconfigure(1, weight=1)
        self.dinov2_frame.grid_columnconfigure(0, weight=1)
        self.dinov2_header = tk.Label(self.dinov2_frame, text="DINOv2", bg=panel, fg=text, font=("Segoe UI", 11, "bold"), anchor="w")
        self.dinov2_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        self.dinov2_table = self._build_rows_panel(self.dinov2_frame, columns=("Attribute", "Best / Top3"))

        self.meili_frame = tk.Frame(main, bg=panel, bd=1, relief="solid")
        self.meili_frame.grid(row=1, column=1, columnspan=3, sticky="nsew", padx=4, pady=(4, 0))
        self.meili_frame.grid_propagate(True)
        self.meili_frame.grid_rowconfigure(0, weight=0)
        self.meili_frame.grid_rowconfigure(1, weight=1)
        self.meili_frame.grid_columnconfigure(0, weight=1)
        self.meili_header = tk.Label(self.meili_frame, text="Meilisearch matches", bg=panel, fg=text, font=("Segoe UI", 11, "bold"), anchor="w")
        self.meili_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))

        self.meili_canvas = tk.Canvas(self.meili_frame, bg="#ffffff", highlightthickness=0)
        self.meili_canvas.grid(row=1, column=0, sticky="nsew", padx=(10, 0), pady=(0, 8))

        self.meili_xscroll = tk.Scrollbar(self.meili_frame, orient="horizontal", command=self.meili_canvas.xview)
        self.meili_xscroll.grid(row=2, column=0, sticky="ew", padx=(10, 0), pady=(0, 8))
        self.meili_yscroll = tk.Scrollbar(self.meili_frame, orient="vertical", command=self.meili_canvas.yview)
        self.meili_yscroll.grid(row=1, column=1, sticky="ns", pady=(0, 8))
        self.meili_canvas.configure(xscrollcommand=self.meili_xscroll.set, yscrollcommand=self.meili_yscroll.set)

        self.meili_table = tk.Frame(self.meili_canvas, bg="#ffffff")
        self.meili_table_id = self.meili_canvas.create_window((0, 0), window=self.meili_table, anchor="nw")
        self.meili_table.grid_columnconfigure(tuple(range(7)), weight=1)
        self.meili_table.grid_propagate(True)

        self.meili_canvas.bind("<Configure>", self._sync_meili_canvas)
        self.meili_table.bind("<Configure>", self._sync_meili_canvas)

        self.status = tk.Label(self, text="", anchor="w", bg=bg, fg=muted, font=("Segoe UI", 10, "bold"))
        self.status.pack(fill="x", padx=14, pady=(0, 10))

        if self.image_names:
            self.on_image_change()

    def _build_rows_panel(self, parent: tk.Misc, columns: tuple[str, ...]):
        frame = tk.Frame(
            parent,
            bg="#ffffff",
        )
        frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=3)
        frame.grid_rowconfigure(0, weight=0)
        frame.grid_rowconfigure(1, weight=1)

        header_row = 0
        for idx, column in enumerate(columns):
            header = tk.Label(frame, text=column, bg="#eef4ff", fg="#1d2433", font=("Segoe UI", 9, "bold"), anchor="w")
            header.grid(row=header_row, column=idx, sticky="ew", padx=(0 if idx == 0 else 8, 8), pady=(0, 6))

        canvas = tk.Canvas(frame, bg="#ffffff", highlightthickness=0)
        canvas.grid(row=1, column=0, columnspan=2, sticky="nsew")

        body = tk.Frame(canvas, bg="#ffffff")
        body_window = canvas.create_window((0, 0), window=body, anchor="nw")

        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=3)

        vsb = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        vsb.grid(row=1, column=2, sticky="ns")
        canvas.configure(yscrollcommand=vsb.set)

        def sync_body_to_canvas(_event=None):
            canvas_width = max(canvas.winfo_width(), 1)
            canvas_height = max(canvas.winfo_height(), 1)
            body_height = max(int(canvas_height * 0.5), body.winfo_reqheight())
            canvas.itemconfigure(body_window, width=canvas_width, height=body_height)
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=bbox)

        canvas.bind("<Configure>", sync_body_to_canvas)
        body.bind("<Configure>", sync_body_to_canvas)
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        return body

    def _replace_rows(self, target: tk.Misc, rows: list[tuple[str, str]]):
        for widget in target.winfo_children():
            widget.destroy()

        target.grid_columnconfigure(0, weight=1, minsize=100)
        target.grid_columnconfigure(1, weight=3, minsize=240)

        if not rows:
            empty = tk.Label(target, text="No data", bg="#ffffff", fg="#5d6982", font=("Segoe UI", 10))
            empty.grid(row=0, column=0, sticky="w", padx=8, pady=8)
            return

        target.update_idletasks()
        value_wraplength = max(260, target.winfo_width() - 170)

        for idx, (label, value) in enumerate(rows):
            target.grid_rowconfigure(idx, weight=1, uniform="attr-rows")
            attr_label = tk.Label(target, text=label, bg="#ffffff", fg="#1d2433", font=("Segoe UI", 9, "bold"), anchor="w", justify="left")
            attr_label.grid(row=idx, column=0, sticky="nsew", padx=(8, 10), pady=4)

            value_bg = self._value_background(value)
            value_label = tk.Label(
                target,
                text=value,
                bg=value_bg,
                fg="#24314d",
                font=("Segoe UI", 9),
                anchor="w",
                justify="left",
                wraplength=value_wraplength,
                padx=6,
                pady=3,
                bd=1,
                relief="solid",
            )
            value_label.grid(row=idx, column=1, sticky="nsew", padx=(0, 8), pady=4)

    def _value_background(self, value: str) -> str:
        if not value or value == "-":
            return "#f5f7fb"
        if "=" in value:
            try:
                score_tokens = [part.split("=")[-1] for part in value.split(";") if "=" in part]
                if not score_tokens:
                    return "#f7fafc"
                scores = [float(token.strip()) for token in score_tokens if token.strip()]
                if not scores:
                    return "#f7fafc"
                highest = max(scores)
                if highest >= 0.85:
                    return "#e6f8ec"
                if highest >= 0.60:
                    return "#fff7db"
                return "#fdecea"
            except ValueError:
                pass
        return "#f7fafc"

    def show_previous_image(self):
        if not self.image_names:
            return
        current = self.current_image.get()
        if not current:
            self.current_image.set(self.image_names[0])
            self.on_image_change()
            return
        index = self.image_names.index(current)
        if index > 0:
            self.current_image.set(self.image_names[index - 1])
            self.on_image_change()

    def show_next_image(self):
        if not self.image_names:
            return
        current = self.current_image.get()
        if not current:
            self.current_image.set(self.image_names[0])
            self.on_image_change()
            return
        index = self.image_names.index(current)
        if index < len(self.image_names) - 1:
            self.current_image.set(self.image_names[index + 1])
            self.on_image_change()

    def on_image_change(self, event=None):
        filename = self.current_image.get()
        if not filename:
            return

        self.update_counter_label(filename)
        product_id = self.ground_truth.get(filename)
        record = self.product_lookup.get(str(product_id), {})
        image_path = self.images_dir / filename

        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((380, 380))
            photo = ImageTk.PhotoImage(image)
            self.preview_canvas.configure(image=photo, text="")
            self.preview_canvas.image = photo
        except Exception as exc:
            self.preview_canvas.configure(text=f"Could not load image:\n{exc}", fg="#7c3b3b", font=("Segoe UI", 11, "bold"))
            self.preview_canvas.image = None

        self._replace_rows(self.actual_table, [("Product ID", str(record.get("id", "-"))), ("Brand", str(record.get("brandName", "-"))), ("Product", str(record.get("productName", "-")))] + build_attribute_rows_for_record(record))

        siglip = self.siglip_predictions.get(filename, {})
        self._replace_rows(self.siglip_table, build_prediction_rows(siglip.get("predicted", {})))

        dinov2 = self.dinov2_predictions.get(filename, {})
        self._replace_rows(self.dinov2_table, build_prediction_rows(dinov2.get("predicted", {})))

        siglip_query = build_query_from_prediction(siglip.get("predicted", {}))
        dinov2_query = build_query_from_prediction(dinov2.get("predicted", {}))
        siglip_hits = fetch_meilisearch(self.host, self.api_key, self.index, siglip_query, limit=5)
        dinov2_hits = fetch_meilisearch(self.host, self.api_key, self.index, dinov2_query, limit=5)

        self._replace_meili_rows(
            [
                ("SigLIP2 query", siglip_query or "empty query", "-", "-", "-", "-", "-"),
                *build_search_hit_rows(siglip_hits),
                ("DINOv2 query", dinov2_query or "empty query", "-", "-", "-", "-", "-"),
                *build_search_hit_rows(dinov2_hits),
            ]
        )

        self.status.config(text=f"Image {self.image_names.index(filename) + 1}/{len(self.image_names)} • Product ID: {product_id or 'n/a'}")

    def _sync_meili_canvas(self, event=None):
        if not self.meili_canvas.winfo_exists():
            return
        canvas_width = max(self.meili_canvas.winfo_width(), 900)
        table_width = self.meili_table.winfo_reqwidth()
        self.meili_canvas.itemconfigure(self.meili_table_id, width=max(canvas_width, table_width))
        bbox = self.meili_canvas.bbox("all")
        if bbox:
            self.meili_canvas.configure(scrollregion=bbox)

    def _replace_meili_rows(self, rows: list[tuple[str, str, str, str, str, str, str]]):
        for widget in self.meili_table.winfo_children():
            widget.destroy()

        columns = ("Rank", "Product ID", "Location", "Application", "Category", "Material", "Color")
        wrap_lengths = [90, 220, 220, 240, 220, 220, 200]

        for idx, column in enumerate(columns):
            header = tk.Label(self.meili_table, text=column, bg="#eef4ff", fg="#1d2433", font=("Segoe UI", 9, "bold"), anchor="w")
            header.grid(row=0, column=idx, sticky="ew", padx=4, pady=(0, 6))
            self.meili_table.grid_columnconfigure(idx, weight=0, minsize=wrap_lengths[idx])

        for r_idx, row in enumerate(rows, start=1):
            for c_idx, value in enumerate(row):
                cell = tk.Label(
                    self.meili_table,
                    text=str(value),
                    bg="#ffffff" if (r_idx % 2 == 1) else "#fafcff",
                    fg="#24314d",
                    font=("Segoe UI", 9),
                    anchor="w",
                    justify="left",
                    wraplength=wrap_lengths[c_idx],
                    padx=4,
                    pady=2,
                )
                cell.grid(row=r_idx, column=c_idx, sticky="nsew", padx=4, pady=2)

        self.meili_table.grid_rowconfigure(0, weight=0)
        for r_idx in range(1, len(rows) + 2):
            self.meili_table.grid_rowconfigure(r_idx, weight=0)

        self.meili_table.update_idletasks()
        self._sync_meili_canvas()

    def update_counter_label(self, filename: str):
        if not self.image_names:
            self.counter_label.config(text="No images available")
            return
        current_index = self.image_names.index(filename) + 1 if filename in self.image_names else 0
        self.counter_label.config(text=f"Image {current_index}/{len(self.image_names)}")

    def format_actual_record(self, record: dict[str, Any]) -> str:
        if not record:
            return "No product record found"
        lines = [
            f"Product ID: {record.get('id')}",
            f"Brand: {record.get('brandName', '-')}",
            f"Product: {record.get('productName', '-')}",
            "",
            "Actual attributes:",
            f"  location: {product_field_summary(record, 'location')}",
            f"  application: {product_field_summary(record, 'application')}",
            f"  productCategory: {product_field_summary(record, 'productCategory')}",
            f"  materialType: {product_field_summary(record, 'materialType')}",
            f"  color: {product_field_summary(record, 'color')}",
        ]
        return "\n".join(lines)

    def format_hits(self, hits: list[dict[str, Any]]) -> str:
        if not hits:
            return "  none"
        lines = []
        for hit in hits[:5]:
            if "error" in hit:
                lines.append(f"  ERROR: {hit['error']}")
                continue
            product_id = hit.get("product_id") or hit.get("id") or "unknown"
            location = ", ".join(as_list(hit.get("location"))) or "-"
            application = ", ".join(as_list(hit.get("application"))) or "-"
            product_category = ", ".join(as_list(hit.get("product_category"))) or "-"
            material_type = ", ".join(as_list(hit.get("material_type"))) or "-"
            color = ", ".join(as_list(hit.get("color"))) or "-"
            lines.append(
                "  "
                + " | ".join(
                    [
                        f"product_id={product_id}",
                        f"location={location}",
                        f"application={application}",
                        f"productCategory={product_category}",
                        f"materialType={material_type}",
                        f"color={color}",
                    ]
                )
            )
        return "\n".join(lines)


def main() -> int:
    app = PredictionViewer()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
