#!/usr/bin/env python3
"""Index HSI-only texture vectors into Meilisearch."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import urlparse

import numpy as np
from PIL import Image

DEFAULT_PRODUCT_EXPORT = "texture/lightbeans_main_page_textures.json"
DEFAULT_TEXTURE_DIR = "texture/main_page"
DEFAULT_FILTER_METADATA_EXPORT = "data/platform-products-production-v2-en.json"
MEILI_INDEX = "lightbeans-texture-hsi"


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


def _is_retryable_network_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "eof occurred in violation of protocol" in message or "connection reset" in message or "timed out" in message


def http_json(
    method: str,
    url: str,
    payload: dict | list | None = None,
    api_key: str | None = None,
    retries: int = 3,
    retry_delay_seconds: float = 1.5,
):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    attempts = max(1, retries + 1)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
        except (error.URLError, ssl.SSLError, OSError) as exc:
            last_exc = exc
            if attempt >= attempts or not _is_retryable_network_error(exc):
                raise RuntimeError(f"Could not reach Meilisearch at {url}: {exc}") from exc
            time.sleep(retry_delay_seconds * attempt)
    raise RuntimeError(f"Could not reach Meilisearch at {url}: {last_exc}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_product_export(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [item for item in data if isinstance(item, dict)]


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def normalize_product_url_key(url: Any) -> str | None:
    if not isinstance(url, str):
        return None
    raw = url.strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    path = parsed.path or raw
    marker = "/texture/"
    idx = path.find(marker)
    if idx < 0:
        return None
    suffix = path[idx + len(marker) :].strip("/")
    parts = [part.strip().lower() for part in suffix.split("/") if part.strip()]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def extract_texture_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None

    marker = "/textures/"
    if marker in raw:
        suffix = raw.split(marker, 1)[1].strip("/")
        parts = [part for part in suffix.split("/") if part]
        if parts:
            return parts[0].lower()

    parts = [part for part in raw.strip("/").split("/") if part]
    if parts:
        return parts[0].lower()
    return None


def load_filter_metadata_export(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = load_json(path)
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def build_product_export_lookup(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for record in records:
        keys: list[str] = []
        product_id = record.get("id")
        if product_id is not None:
            keys.append(f"id:{str(product_id)}")

        link_key = normalize_product_url_key(record.get("link"))
        if link_key:
            keys.append(f"url:{link_key}")

        texture_key = extract_texture_id(record.get("image_url"))
        if texture_key:
            keys.append(f"tex:{texture_key}")

        for key in keys:
            lookup[key] = record
    return lookup


def resolve_product_export_record(metadata_record: dict[str, Any], product_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    product_id = metadata_record.get("id")
    if product_id is not None:
        match = product_lookup.get(f"id:{str(product_id)}")
        if match is not None:
            return match

    brand_slug = metadata_record.get("brandSlug")
    product_slug = metadata_record.get("productSlug")
    if isinstance(brand_slug, str) and isinstance(product_slug, str):
        key = f"url:{brand_slug.strip().lower()}/{product_slug.strip().lower()}"
        match = product_lookup.get(key)
        if match is not None:
            return match

    texture_key = extract_texture_id(metadata_record.get("previewPath"))
    if texture_key:
        match = product_lookup.get(f"tex:{texture_key}")
        if match is not None:
            return match

    return {}


def build_product_url(brand_slug: Any, product_slug: Any) -> str | None:
    if not isinstance(brand_slug, str) or not isinstance(product_slug, str):
        return None
    brand = brand_slug.strip().strip("/")
    product = product_slug.strip().strip("/")
    if not brand or not product:
        return None
    return f"https://lightbeans.com/en/texture/{brand}/{product}"


def load_mapping(mapping_path: Path | None) -> dict[str, Any]:
    if mapping_path is None or not mapping_path.exists():
        return {"files": {}}
    payload = load_json(mapping_path)
    if not isinstance(payload, dict):
        return {"files": {}}
    files = payload.get("files", {})
    if isinstance(files, dict):
        return {"files": files}
    return {"files": {}}


def normalize_preview_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    preview_path = value.strip().lstrip("/")
    if preview_path.endswith("/01.png"):
        preview_path = preview_path[: -len("/01.png")] + "/plane-1800.webp"
    return preview_path or None


def resolve_texture_path(product: dict[str, Any], texture_dir: Path, mapping_path: Path | None) -> Path | None:
    image_name = product.get("image_name")
    if isinstance(image_name, str) and image_name.strip():
        for candidate_dir in [texture_dir, texture_dir / "main_page", texture_dir.parent / "main_page"]:
            candidate = candidate_dir / image_name.strip()
            if candidate.exists():
                return candidate

    image_url = product.get("image_url")
    if isinstance(image_url, str) and image_url.strip():
        filename = image_url.rstrip("/").split("/")[-1]
        if filename:
            for candidate_dir in [texture_dir, texture_dir / "main_page", texture_dir.parent / "main_page"]:
                candidate = candidate_dir / filename
                if candidate.exists():
                    return candidate

    product_id = product.get("id")
    if product_id is not None:
        mapping = load_mapping(mapping_path)
        files = mapping.get("files", {})
        for relative_path, meta in files.items():
            if not isinstance(meta, dict):
                continue
            product_ids = meta.get("product_ids", [])
            if str(product_id) in [str(pid) for pid in product_ids]:
                candidate = texture_dir / relative_path
                if candidate.exists():
                    return candidate

    preview_path = normalize_preview_path(product.get("previewPath"))
    if not preview_path:
        return None
    candidate = texture_dir / preview_path
    if candidate.exists():
        return candidate
    return None


def wait_for_task(host: str, api_key: str | None, task_uid: int, timeout_seconds: int = 600, poll_seconds: float = 0.5) -> None:
    start = time.time()
    endpoint = f"{host.rstrip('/')}/tasks/{task_uid}"
    while True:
        task = http_json("GET", endpoint, api_key=api_key)
        status = task.get("status")
        if status == "succeeded":
            return
        if status == "failed":
            raise RuntimeError(f"Meilisearch task {task_uid} failed: {task.get('error')}")
        if time.time() - start > timeout_seconds:
            raise RuntimeError(f"Timed out waiting for Meilisearch task {task_uid} after {timeout_seconds}s.")
        time.sleep(poll_seconds)


def extract_task_uid(response: dict[str, Any], operation_name: str) -> int:
    task_uid = response.get("taskUid")
    if not isinstance(task_uid, int):
        raise RuntimeError(f"{operation_name} did not return a valid taskUid: {response}")
    return task_uid


def ensure_index(host: str, api_key: str | None, index_name: str, vector_dimensions: int | None = None) -> None:
    endpoint = f"{host.rstrip('/')}/indexes"
    payload = {"uid": index_name, "primaryKey": "product_id"}
    try:
        response = http_json("POST", endpoint, payload, api_key)
        wait_for_task(host, api_key, extract_task_uid(response, "create-index"))
    except RuntimeError as exc:
        message = str(exc).lower()
        if "already exists" not in message and "index_already_exists" not in message:
            raise

    settings: dict[str, Any] = {
        "searchableAttributes": [
            "name",
            "link",
            "image_url",
            "image_name",
            "brandName",
            "brand_name",
            "productCategory",
            "materialType",
            "location",
            "application",
            "product_category",
            "material_type",
            "color",
        ],
        "filterableAttributes": [
            "product_id",
            "name",
            "link",
            "image_url",
            "image_name",
            "brandName",
            "brand_name",
            "location",
            "application",
            "productCategory",
            "materialType",
            "product_category",
            "material_type",
            "color",
            "locale",
        ],
        "sortableAttributes": ["product_id"],
        "rankingRules": ["words", "typo", "proximity", "attribute", "sort", "exactness"],
    }
    if vector_dimensions is not None:
        settings["embedders"] = {"default": {"source": "userProvided", "dimensions": vector_dimensions}}

    response = http_json("PATCH", f"{host.rstrip('/')}/indexes/{index_name}/settings", settings, api_key)
    wait_for_task(host, api_key, extract_task_uid(response, "update-settings"))


def sanitize_meilisearch_id(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace(".", "_")
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    text = text.strip("_")
    if not text:
        raise ValueError(f"Could not create a valid Meilisearch document ID from value {value!r}")
    return text[:511]


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vector, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vector / norms


def rgb_to_hsi(image_rgb: np.ndarray) -> np.ndarray:
    rgb = image_rgb.astype(np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    intensity = (r + g + b) / 3.0
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.zeros_like(intensity)
    nonzero = intensity > 1e-8
    saturation[nonzero] = 1.0 - (min_rgb[nonzero] / intensity[nonzero])

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    den = np.where(den == 0, 1e-8, den)
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    hue = np.where(b <= g, theta, (2.0 * np.pi) - theta)
    hue = hue / (2.0 * np.pi)

    return np.stack([hue, saturation, intensity], axis=-1)


def compute_hsi_histogram(image_path: Path, h_bins: int, s_bins: int, i_bins: int) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    try:
        rgb = np.asarray(image, dtype=np.uint8)
        hsi = rgb_to_hsi(rgb)
        h = hsi[..., 0].ravel()
        s = hsi[..., 1].ravel()
        i = hsi[..., 2].ravel()

        hist_h, _ = np.histogram(h, bins=h_bins, range=(0.0, 1.0))
        hist_s, _ = np.histogram(s, bins=s_bins, range=(0.0, 1.0))
        hist_i, _ = np.histogram(i, bins=i_bins, range=(0.0, 1.0))
        feature = np.concatenate([hist_h, hist_s, hist_i]).astype(np.float32)
        feature = l2_normalize(feature.reshape(1, -1))[0]
        return [float(v) for v in feature.tolist()]
    finally:
        image.close()


def build_features(record: dict[str, Any]) -> list[str]:
    output: list[str] = []
    for field in ["name", "link", "image_url", "image_name"]:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            output.append(value.strip())
    return output


def build_document(
    product: dict[str, Any],
    image_path: Path,
    vector: list[float],
    metadata_record: dict[str, Any],
) -> dict[str, Any]:
    raw_product_id = product.get("id") or product.get("image_name") or product.get("link") or image_path.stem
    product_id = sanitize_meilisearch_id(raw_product_id)
    merged = dict(metadata_record)
    merged.update(product)
    link = merged.get("link") or build_product_url(merged.get("brandSlug"), merged.get("productSlug"))
    image_url = merged.get("image_url")
    if not image_url and isinstance(merged.get("previewPath"), str):
        preview_path = normalize_preview_path(merged.get("previewPath"))
        if preview_path:
            image_url = f"https://cdn.lightbeans.com/textures/{preview_path.lstrip('/')}"
    merged["link"] = link
    merged["image_url"] = image_url or ""
    return {
        "product_id": product_id,
        "id": product_id,
        "name": merged.get("name") or merged.get("productName"),
        "link": link,
        "image_url": image_url or "",
        "image_name": merged.get("image_name") or image_path.name,
        "image_path": image_path.as_posix(),
        "features": build_features(merged),
        "brandName": merged.get("brandName") or merged.get("brand_name"),
        "brand_name": merged.get("brandName") or merged.get("brand_name"),
        "location": as_list(merged.get("location")),
        "application": as_list(merged.get("application")),
        "productCategory": as_list(merged.get("productCategory") or merged.get("product_category")),
        "materialType": as_list(merged.get("materialType") or merged.get("material_type")),
        "product_category": as_list(merged.get("productCategory") or merged.get("product_category")),
        "material_type": as_list(merged.get("materialType") or merged.get("material_type")),
        "color": as_list(merged.get("color")),
        "locale": merged.get("locale"),
        "embedding": vector,
        "_vectors": {"default": vector},
    }


def upload_documents(
    host: str,
    api_key: str | None,
    index_name: str,
    documents: list[dict[str, Any]],
    batch_size: int = 100,
    trace: bool = False,
) -> None:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/documents"
    total = len(documents)
    if total == 0:
        return
    for start in range(0, total, batch_size):
        batch = documents[start : start + batch_size]
        if trace:
            end = min(start + len(batch), total)
            print(f"[trace] Uploading batch {start + 1}-{end} of {total} documents.")
        response = http_json("POST", endpoint, batch, api_key, retries=5, retry_delay_seconds=2.0)
        wait_for_task(host, api_key, extract_task_uid(response, "upload-documents"))


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    env_values = load_env(project_root / ".env")
    parser = argparse.ArgumentParser(description="Index HSI-only vectors into Meilisearch.")
    parser.add_argument(
        "--product-export",
        type=Path,
        default=Path(DEFAULT_PRODUCT_EXPORT),
        help="Optional product export JSON used to enrich image_name/link/image_url matching.",
    )
    parser.add_argument("--textures-dir", type=Path, default=Path(DEFAULT_TEXTURE_DIR), help="Directory containing downloaded texture files.")
    parser.add_argument(
        "--filter-metadata-export",
        type=Path,
        default=Path(DEFAULT_FILTER_METADATA_EXPORT),
        help="Optional product metadata export used to enrich filter attributes.",
    )
    parser.add_argument("--mapping-json", type=Path, default=None, help="Optional file-to-product mapping JSON for legacy texture exports.")
    parser.add_argument("--host", default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700", help="Meilisearch host.")
    parser.add_argument("--key", default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"), help="Meilisearch API key.")
    parser.add_argument("--index", default=env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or MEILI_INDEX, help="Target Meilisearch index name.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N products (0 = all).")
    parser.add_argument("--h-bins", type=int, default=36, help="Hue histogram bins.")
    parser.add_argument("--s-bins", type=int, default=16, help="Saturation histogram bins.")
    parser.add_argument("--i-bins", type=int, default=16, help="Intensity histogram bins.")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of documents per Meilisearch upload batch.")
    parser.add_argument("--trace", dest="trace", action="store_true", help="Print progression traces while indexing.")
    parser.add_argument("--no-trace", dest="trace", action="store_false", help="Disable progression traces while indexing.")
    parser.set_defaults(trace=True)
    parser.add_argument("--trace-every", type=int, default=100, help="Emit progress every N metadata records when --trace is enabled.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.filter_metadata_export.exists():
        raise FileNotFoundError(f"Filter metadata export not found: {args.filter_metadata_export}")
    if not args.textures_dir.exists():
        raise FileNotFoundError(f"Textures directory not found: {args.textures_dir}")
    if args.h_bins <= 0 or args.s_bins <= 0 or args.i_bins <= 0:
        raise ValueError("All histogram bin counts must be > 0.")
    if args.trace_every <= 0:
        raise ValueError("--trace-every must be > 0.")

    metadata_records = load_filter_metadata_export(args.filter_metadata_export)
    product_records: list[dict[str, Any]] = []
    if args.product_export.exists():
        product_records = load_product_export(args.product_export)
    product_lookup = build_product_export_lookup(product_records)

    if args.limit > 0:
        metadata_records = metadata_records[: args.limit]

    total_records = len(metadata_records)
    if args.trace:
        print(
            "[trace] Starting HSI indexing "
            f"(metadata={total_records}, product_export={len(product_records)}, limit={args.limit or 'all'})."
        )

    documents: list[dict[str, Any]] = []
    skipped_missing_texture = 0
    skipped_missing_id = 0
    for index, metadata_record in enumerate(metadata_records, start=1):
        base_record = resolve_product_export_record(metadata_record, product_lookup)
        resolved_record = dict(metadata_record)
        resolved_record.update(base_record)

        product_id = (
            resolved_record.get("id")
            or resolved_record.get("image_name")
            or resolved_record.get("link")
            or resolved_record.get("image_url")
            or resolved_record.get("previewPath")
        )
        if product_id is None:
            skipped_missing_id += 1
            if args.trace:
                print(f"[trace] Skipped {index}/{total_records}: missing id/link/image reference.")
            continue
        if "id" not in resolved_record and product_id is not None:
            resolved_record["id"] = sanitize_meilisearch_id(product_id)

        texture_path = resolve_texture_path(resolved_record, args.textures_dir, args.mapping_json)
        if texture_path is None or not texture_path.exists():
            skipped_missing_texture += 1
            if args.trace:
                current_id = resolved_record.get("id") or product_id
                print(f"[trace] Skipped {index}/{total_records}: texture not found for '{current_id}'.")
            continue

        vector = compute_hsi_histogram(texture_path, args.h_bins, args.s_bins, args.i_bins)
        documents.append(build_document(resolved_record, texture_path, vector, metadata_record))
        if args.trace and (index == 1 or index % args.trace_every == 0 or index == total_records):
            print(
                f"[trace] Progress {index}/{total_records}: indexed={len(documents)}, "
                f"skipped_texture={skipped_missing_texture}, skipped_missing_id={skipped_missing_id}."
            )

    vector_dimensions = len(documents[0]["embedding"]) if documents else None
    ensure_index(args.host, args.key, args.index, vector_dimensions=vector_dimensions)
    upload_documents(args.host, args.key, args.index, documents, batch_size=args.batch_size, trace=args.trace)
    print(f"Indexed {len(documents)} HSI-only vectors in Meilisearch index '{args.index}'.")
    print(f"Skipped {skipped_missing_texture} products without a usable texture file.")
    if skipped_missing_id:
        print(f"Skipped {skipped_missing_id} records without id/link/image reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
