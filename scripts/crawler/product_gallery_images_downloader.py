#!/usr/bin/env python3
"""Download product gallery images from the Lightbeans platform export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ROOT_URL = "https://cdn.lightbeans.com/"


def iter_render_paths(record: Any) -> list[str]:
    if not isinstance(record, dict):
        return []

    renders = record.get("renders")
    if renders is None:
        return []
    if isinstance(renders, str):
        renders = [renders]
    if not isinstance(renders, list):
        return []

    paths: list[str] = []
    for value in renders:
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())
    return paths


def resolve_download_url(value: str) -> str:
    if value.startswith(("http://", "https://")):
        return value
    return urljoin(ROOT_URL, value.lstrip("/"))


def filename_from_render_path(value: str) -> str | None:
    cleaned = value.strip()
    if not cleaned:
        return None

    marker = "product-gallery-images/"
    if marker in cleaned:
        cleaned = cleaned.split(marker, 1)[1]
    elif cleaned.startswith("/"):
        cleaned = cleaned.lstrip("/")

    if not cleaned:
        return None

    return cleaned.split("?", 1)[0].split("#", 1)[0]


def download_file(url: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(request, timeout=30) as response:
            content_type = response.headers.get_content_type()
            data = response.read()
    except Exception as exc:  # pragma: no cover - exercised via CLI
        print(f"Failed to download {url}: {exc}")
        return False

    if not data:
        print(f"Empty response for {url}")
        return False

    if destination.exists() and destination.stat().st_size == len(data):
        return True

    destination.write_bytes(data)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Lightbeans product gallery images from a JSON export.")
    parser.add_argument("input", help="Path to the platform JSON export (for example data/platform-products-production-v2-en.json)")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="images",
        help="Directory where images should be saved (default: images)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist to make reruns faster",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {input_path}, got {type(records).__name__}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    seen: set[str] = set()

    for record in records:
        for render_path in iter_render_paths(record):
            filename = filename_from_render_path(render_path)
            if not filename:
                continue

            if filename in seen:
                continue
            seen.add(filename)

            destination = output_dir / filename
            if args.skip_existing and destination.exists():
                skipped += 1
                continue

            url = resolve_download_url(render_path)
            if download_file(url, destination):
                downloaded += 1
            else:
                print(f"Skipped {filename} due to download failure")

    print(f"Downloaded {downloaded} images into {output_dir}")
    if skipped:
        print(f"Skipped {skipped} existing files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
