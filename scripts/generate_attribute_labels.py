#!/usr/bin/env python3
"""Generate a reusable attribute vocabulary JSON from the platform export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIELDS = ["location", "application", "productCategory", "materialType", "color"]


def normalize_value(value: Any) -> str:
    text = str(value).strip()
    return text


def build_attribute_labels(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {field: [] for field in FIELDS}
    seen: dict[str, set[str]] = {field: set() for field in FIELDS}

    for record in records:
        for field in FIELDS:
            raw_values = record.get(field, [])
            if isinstance(raw_values, str):
                raw_values = [raw_values]
            if not isinstance(raw_values, list):
                continue

            for value in raw_values:
                text = normalize_value(value)
                if not text:
                    continue
                if text in seen[field]:
                    continue
                seen[field].add(text)
                labels[field].append(text)

    for field in FIELDS:
        labels[field] = sorted(labels[field])

    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract the unique attribute values used in the product export.")
    parser.add_argument("input", nargs="?", default="data/platform-products-production-v2-en.json", help="Product export JSON file")
    parser.add_argument("-o", "--output", default="attribute_labels.json", help="Output JSON containing attribute labels")
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {input_path}")

    labels = build_attribute_labels(data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(labels, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Wrote unique labels for {len(FIELDS)} attributes to {output_path}")
    for field in FIELDS:
        print(f"  {field}: {len(labels[field])} values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
