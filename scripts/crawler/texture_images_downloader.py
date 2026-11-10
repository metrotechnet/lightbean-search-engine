#!/usr/bin/env python3
"""Download texture images using previewPath and build a file->product mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

BASE_TEXTURE_URL = "https://cdn.lightbeans.com/textures/"


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def normalize_preview_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    preview_path = value.strip().lstrip("/")
    if preview_path.endswith("/01.png"):
        preview_path = preview_path[: -len("/01.png")] + "/plane-1800.webp"
    return preview_path or None


def download_file(url: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read()
    except Exception as exc:  # pragma: no cover - exercised via CLI
        print(f"Failed to download {url}: {exc}")
        return False

    if not payload:
        print(f"Empty response for {url}")
        return False

    if destination.exists() and destination.stat().st_size == len(payload):
        return True

    destination.write_bytes(payload)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download texture images from previewPath values.")
    parser.add_argument(
        "input",
        nargs="?",
        default="data/platform-products-production-v2-en.json",
        help="Product export JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="texture",
        help="Directory where texture files will be saved.",
    )
    parser.add_argument(
        "--mapping-output",
        default="texture/texture_product_mapping.json",
        help="Output JSON mapping path.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip already existing files.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    records = load_records(input_path)

    downloaded = 0
    skipped = 0
    failed = 0
    mapping: dict[str, dict[str, Any]] = {}

    for record in records:
        product_id = record.get("id")
        preview_path = normalize_preview_path(record.get("previewPath"))
        if product_id is None or not preview_path:
            continue

        destination = output_dir / preview_path
        relative_file = str(destination.relative_to(output_dir).as_posix())
        url = urljoin(BASE_TEXTURE_URL, preview_path)

        if args.skip_existing and destination.exists():
            skipped += 1
            status = "skipped_existing"
        else:
            ok = download_file(url, destination)
            if ok:
                downloaded += 1
                status = "downloaded"
            else:
                failed += 1
                status = "failed"

        existing = mapping.get(relative_file)
        if existing is None:
            mapping[relative_file] = {
                "product_ids": [str(product_id)],
                "preview_path": preview_path,
                "url": url,
                "status": status,
            }
        else:
            if str(product_id) not in existing["product_ids"]:
                existing["product_ids"].append(str(product_id))
            if existing.get("status") != "downloaded" and status == "downloaded":
                existing["status"] = "downloaded"

    summary = {
        "base_url": BASE_TEXTURE_URL,
        "total_records": len(records),
        "files_in_mapping": len(mapping),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "failed": failed,
    }
    output = {
        "summary": summary,
        "files": dict(sorted(mapping.items())),
    }

    mapping_output_path = Path(args.mapping_output)
    mapping_output_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Textures downloaded: {downloaded}")
    print(f"Existing skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Mapping written to {mapping_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
