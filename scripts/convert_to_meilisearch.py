#!/usr/bin/env python3
"""Convert product JSON exports into Meilisearch-friendly documents.

Usage:
  python convert_to_meilisearch.py input.json --output meilisearch-products.json
  python convert_to_meilisearch.py platform-products-production-v2-en.json
  python convert_to_meilisearch.py platform-products-production-v2-en.json platform-products-production-v2-fr.json --output meilisearch-products.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def collect_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            strings.extend(collect_strings(item))
    elif isinstance(value, list):
        for item in value:
            strings.extend(collect_strings(item))
    elif isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        if text:
            strings.append(text)
    return strings


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def extract_variant_text(item: Any) -> list[str]:
    values: list[str] = []
    if isinstance(item, dict):
        for key in ("name", "slug", "prefix", "suffix", "displayUnits", "value"):
            if key in item:
                values.extend(collect_strings(item[key]))
        if "characteristicValue" in item:
            values.extend(extract_variant_text(item["characteristicValue"]))
        if "characteristic" in item:
            values.extend(extract_variant_text(item["characteristic"]))
    elif isinstance(item, list):
        for entry in item:
            values.extend(extract_variant_text(entry))
    return values


def build_document(record: dict[str, Any], use_locale_suffix: bool = False) -> dict[str, Any]:
    product_id = record.get("id")
    doc_id = product_id
    if use_locale_suffix:
        locale = record.get("locale")
        doc_id = f"{product_id}-{locale}" if locale else product_id

    attributes = {
        "location": to_list(record.get("location")),
        "application": to_list(record.get("application")),
        "product_category": to_list(record.get("productCategory")),
        "material_type": to_list(record.get("materialType")),
        "color": to_list(record.get("color")),
        "availability_countries": (record.get("availability") or {}).get("countries", []),
    }

    searchable_parts: list[str] = []
    searchable_parts.extend(collect_strings(record.get("productName")))
    searchable_parts.extend(collect_strings(record.get("brandName")))
    searchable_parts.extend(collect_strings(record.get("brandSlug")))
    searchable_parts.extend(collect_strings(record.get("productSlug")))
    searchable_parts.extend(collect_strings(record.get("organizationSlug")))
    searchable_parts.extend(collect_strings(record.get("location")))
    searchable_parts.extend(collect_strings(record.get("application")))
    searchable_parts.extend(collect_strings(record.get("productCategory")))
    searchable_parts.extend(collect_strings(record.get("materialType")))
    searchable_parts.extend(collect_strings(record.get("color")))

    for characteristic in to_list(record.get("characteristics")):
        if not isinstance(characteristic, dict):
            continue
        searchable_parts.extend(extract_variant_text(characteristic))
        searchable_parts.extend(collect_strings(characteristic.get("characteristic")))
        for value in to_list(characteristic.get("values")):
            if isinstance(value, dict):
                searchable_parts.extend(extract_variant_text(value))
                searchable_parts.extend(collect_strings(value.get("characteristicValue")))
                searchable_parts.extend(collect_strings(value.get("prefix")))
                searchable_parts.extend(collect_strings(value.get("suffix")))

    document = {
        "id": doc_id,
        "product_id": product_id,
        "tid": record.get("tid"),
        "locale": record.get("locale"),
        "brand_name": record.get("brandName"),
        "brand_slug": record.get("brandSlug"),
        "product_name": record.get("productName"),
        "product_slug": record.get("productSlug"),
        "organization_slug": record.get("organizationSlug"),
        "featured": bool(record.get("featured", False)),
        "publication_state": record.get("publicationState"),
        "product_page_v2": bool(record.get("productPageV2", False)),
        "physical_dimensions": record.get("physicalDimensions"),
        "preview_path": record.get("previewPath"),
        "renders": record.get("renders", []),
        "created_at": record.get("createdAt"),
        "availability_countries": (record.get("availability") or {}).get("countries", []),
        "location": record.get("location", []),
        "application": record.get("application", []),
        "product_category": record.get("productCategory", []),
        "material_type": record.get("materialType", []),
        "color": record.get("color", []),
        "attributes": attributes,
        "search_text": " ".join(unique_preserve_order(searchable_parts)),
    }

    return document


def load_records(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, list):
            raise ValueError(f"{file_path} does not contain a JSON array of records")

        for item in data:
            if isinstance(item, dict):
                records.append(item)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert product JSON exports into Meilisearch-compatible JSON documents.")
    parser.add_argument("inputs", nargs="+", help="One or more JSON input files to convert")
    parser.add_argument("-o", "--output", default="meilisearch-documents.json", help="Path to the output JSON file")
    args = parser.parse_args()

    records = load_records(args.inputs)
    ids = [record.get("id") for record in records if isinstance(record, dict)]
    use_locale_suffix = len(ids) != len(set(ids))
    documents = [build_document(record, use_locale_suffix=use_locale_suffix) for record in records]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(documents, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Converted {len(documents)} documents to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
