#!/usr/bin/env python3
"""Build a filename -> product_id ground-truth mapping from the platform export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def normalize_filename(render_value: str) -> str | None:
    value = render_value.strip()
    if not value:
        return None

    if "product-gallery-images/" in value:
        value = value.split("product-gallery-images/", 1)[1]
    value = value.lstrip("/")
    value = value.split("?", 1)[0].split("#", 1)[0]
    return value or None


def build_ground_truth(records: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for record in records:
        product_id = record.get("id")
        if not isinstance(product_id, (str, int)):
            continue
        product_id = str(product_id)

        renders = record.get("renders") or []
        if not isinstance(renders, list):
            continue

        for render in renders:
            if not isinstance(render, str):
                continue
            filename = normalize_filename(render)
            if not filename:
                continue
            if filename in mapping and mapping[filename] != product_id:
                raise ValueError(f"Filename '{filename}' maps to multiple product IDs: {mapping[filename]} and {product_id}")
            mapping[filename] = product_id

    return dict(sorted(mapping.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a filename-to-product-id map from the platform export.")
    parser.add_argument("input", nargs="?", default="data/platform-products-production-v2-en.json", help="Input product export file")
    parser.add_argument("-o", "--output", default="ground_truth.json", help="Output JSON file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {input_path}")

    ground_truth = build_ground_truth(data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(ground_truth, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Wrote {len(ground_truth)} filename/product associations to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
