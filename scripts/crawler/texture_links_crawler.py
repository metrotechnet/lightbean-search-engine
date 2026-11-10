#!/usr/bin/env python3
"""Extract product cards from the main Lightbeans textures page.

This keeps the scrape on the main page, finds every link whose href points to a
product detail page, then reads the nested <img> alt text and source URL. The
image URL is rewritten to the 1800px variant as requested.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright

DEFAULT_URL = "https://lightbeans.com/fr/textures"
DEFAULT_OUTPUT = Path("texture/lightbeans_main_page_textures.json")
DEFAULT_IMAGES_DIR = Path("texture/main_page")
DEFAULT_INPUT_JSON = DEFAULT_OUTPUT


def full_url(value: str) -> str:
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("//"):
        return "https:" + value
    return urljoin("https://lightbeans.com", value)


def normalize_image_url(value: str) -> str:
    if not value:
        return ""
    url = full_url(value)
    return re.sub(r"/plane-600\.(?:webp|wepp)", "/plane-1800.webp", url, flags=re.IGNORECASE)


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip())
    cleaned = cleaned.strip("-_.")
    return cleaned or "texture"


def download_image(url: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return True
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://lightbeans.com/fr/textures"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
        if not payload:
            return False
        destination.write_bytes(payload)
        return True
    except Exception as exc:  # pragma: no cover - logging only
        print(f"Failed to download {url}: {exc}")
        return False


def download_from_json(json_path: Path, images_dir: Path) -> list[dict[str, str]]:
    with json_path.open("r", encoding="utf-8") as fh:
        items = json.load(fh)
    if not isinstance(items, list):
        raise ValueError(f"Expected a JSON array in {json_path}")

    successful = 0
    failed = 0
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        if not item.get("name"):
            continue
        item["image_name"] = item.get("image_name") or f"{index:03d}-{sanitize_filename(item['name'])}.webp"
        image_url = item.get("image_url")
        if not image_url:
            failed += 1
            continue
        destination = images_dir / item["image_name"]
        if destination.exists():
            successful += 1
            continue
        if download_image(image_url, destination):
            successful += 1
        else:
            failed += 1
            item["image_url"] = ""
            item["image_name"] = ""

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"Downloaded {successful} images; skipped {failed} blocked/invalid CDN entries.")
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract names and links from the main Lightbeans textures page.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Lightbeans textures page URL.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help="Existing JSON file containing product name/link/image_url entries to download without re-scraping.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON file to write results to.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory where the 1800px images will be saved.",
    )
    parser.add_argument(
        "--max-scrolls",
        type=int,
        default=200,
        help="Maximum scroll attempts; when no new products load for two consecutive steps, the scraper stops automatically.",
    )
    parser.add_argument(
        "--download-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download each 1800px preview after extraction. Use --no-download-images to skip downloads.",
    )
    parser.add_argument(
        "--from-json",
        action="store_true",
        help="Read the existing JSON file and download images from it without extracting product links from the page.",
    )
    args = parser.parse_args()

    if args.from_json:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.images_dir.mkdir(parents=True, exist_ok=True)
        items = download_from_json(args.input_json, args.images_dir)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(items, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(f"Downloaded images from {args.input_json}")
        print(f"Saved JSON to {args.output}")
        print(f"Saved images to {args.images_dir}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.download_images:
        args.images_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1200})
        page.goto(args.url, timeout=120000, wait_until="domcontentloaded")

        last_count = 0
        stable_steps = 0
        for step in range(1, args.max_scrolls + 1):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(1200)
            count = page.locator('a[href*="/fr/texture/"]').count()
            print(f"scroll {step}: {count} links")
            if count == last_count:
                stable_steps += 1
                if stable_steps >= 2:
                    break
            else:
                stable_steps = 0
            last_count = count

        links = page.locator('a[href*="/fr/texture/"]')
        seen_urls: set[str] = set()
        items: list[dict[str, str]] = []

        for idx in range(links.count()):
            link = links.nth(idx)
            href = (link.get_attribute("href") or "").strip()
            if not href:
                continue

            full_href = full_url(href)
            if full_href in seen_urls:
                continue
            seen_urls.add(full_href)

            img = link.locator("img").first
            if img.count() == 0:
                continue

            name = (img.get_attribute("alt") or "").strip()
            src = (img.get_attribute("src") or "").strip()
            if not name or not src:
                continue

            item = {
                "name": name,
                "link": full_href,
                "image_url": normalize_image_url(src),
            }
            item["image_name"] = ""
            items.append(item)

        browser.close()

    for index, item in enumerate(items, start=1):
        item["image_name"] = f"{index:03d}-{sanitize_filename(item['name'])}.webp"
        if not args.download_images:
            continue

        destination = args.images_dir / item["image_name"]
        if destination.exists():
            continue
        download_image(item["image_url"], destination)

    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Extracted {len(items)} product links from {args.url}")
    print(f"Saved JSON to {args.output}")
    if args.download_images:
        print(f"Saved images to {args.images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
