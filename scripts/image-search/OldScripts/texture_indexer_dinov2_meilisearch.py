#!/usr/bin/env python3
"""Index texture-image embeddings from DINOv2 into Meilisearch.

This script loads products from the Lightbeans export, scans a texture-image directory (for example `texture/views`),
matches each image to a product using the source image stem, computes a DINOv2 embedding, and uploads metadata + vector.
The resulting index can be queried by text or nearest-neighbor similarity using the stored vector.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-small"
DEFAULT_PRODUCT_EXPORT = "texture/lightbeans_main_page_textures.json"
DEFAULT_FILTER_METADATA_EXPORT = "data/platform-products-production-v2-en.json"
DEFAULT_TEXTURE_DIR = "texture/views"
MEILI_INDEX = "lightbeans-texture-dinov2-views"
DEFAULT_INDEX = MEILI_INDEX
DEFAULT_MAX_UPLOAD_PAYLOAD_MIB = 90


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


def http_json(method: str, url: str, payload: dict | list | None = None, api_key: str | None = None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach Meilisearch at {url}: {exc}") from exc


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def normalize_preview_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    preview_path = value.strip().lstrip("/")
    if preview_path.endswith("/01.png"):
        preview_path = preview_path[: -len("/01.png")] + "/plane-1800.webp"
    return preview_path or None


def normalize_stem(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).stem.lower()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_product_export(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [item for item in data if isinstance(item, dict)]


def load_filter_metadata_export(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [item for item in data if isinstance(item, dict)]


def build_filter_metadata_lookup(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in records:
        keys: list[str] = []
        product_id = item.get("id")
        if product_id is not None:
            keys.append(f"id:{str(product_id)}")

        brand_slug = item.get("brandSlug")
        product_slug = item.get("productSlug")
        if isinstance(brand_slug, str) and isinstance(product_slug, str):
            keys.append(f"url:{brand_slug.strip().lower()}/{product_slug.strip().lower()}")

        preview_key = extract_texture_id(item.get("previewPath"))
        if preview_key:
            keys.append(f"tex:{preview_key}")

        for key in keys:
            lookup[key] = item
    return lookup


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


def collect_texture_images(texture_dir: Path) -> list[Path]:
    allowed_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    images: list[Path] = []
    for path in sorted(texture_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_ext:
            continue
        if path.stem.lower() == "contact_sheet":
            continue
        images.append(path)
    return images


def collect_view_folders(texture_dir: Path) -> list[tuple[Path, list[Path]]]:
    folder_to_images: dict[Path, list[Path]] = {}
    for image_path in collect_texture_images(texture_dir):
        folder_to_images.setdefault(image_path.parent, []).append(image_path)
    folders = sorted(folder_to_images.keys())
    return [(folder, sorted(folder_to_images[folder])) for folder in folders]


def build_view_folder_lookup(view_folders: list[tuple[Path, list[Path]]]) -> dict[str, tuple[Path, list[Path]]]:
    lookup: dict[str, tuple[Path, list[Path]]] = {}
    for folder_path, view_paths in view_folders:
        lookup[normalize_stem(folder_path.name)] = (folder_path, view_paths)
    return lookup


def resolve_view_folder_for_record(
    record: dict[str, Any],
    folder_lookup: dict[str, tuple[Path, list[Path]]],
) -> tuple[Path, list[Path]] | None:
    candidates: list[str] = []

    image_name = record.get("image_name")
    if isinstance(image_name, str) and image_name.strip():
        candidates.append(normalize_stem(image_name))

    image_url = record.get("image_url")
    if isinstance(image_url, str) and image_url.strip():
        filename = image_url.rstrip("/").split("/")[-1]
        if filename:
            candidates.append(normalize_stem(filename))

    source_image_name = record.get("source_image_name")
    if isinstance(source_image_name, str) and source_image_name.strip():
        candidates.append(normalize_stem(source_image_name))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        folder = folder_lookup.get(candidate)
        if folder is not None:
            return folder
    return None


def build_product_lookup(products: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for product in products:
        product_id = product.get("id")
        if product_id is not None:
            lookup[f"id:{str(product_id)}"] = product

        link_key = normalize_product_url_key(product.get("link"))
        if link_key:
            lookup[f"url:{link_key}"] = product

        image_name = product.get("image_name")
        if isinstance(image_name, str) and image_name.strip():
            lookup.setdefault(f"stem:{normalize_stem(image_name)}", product)

        image_url = product.get("image_url")
        if isinstance(image_url, str) and image_url.strip():
            filename = image_url.rstrip("/").split("/")[-1]
            if filename:
                lookup.setdefault(f"stem:{normalize_stem(filename)}", product)
            texture_key = extract_texture_id(image_url)
            if texture_key:
                lookup[f"tex:{texture_key}"] = product

        preview_path = normalize_preview_path(product.get("previewPath"))
        if preview_path:
            lookup.setdefault(f"stem:{normalize_stem(preview_path)}", product)
            preview_texture_key = extract_texture_id(preview_path)
            if preview_texture_key:
                lookup[f"tex:{preview_texture_key}"] = product
    return lookup


def resolve_filter_metadata(product_record: dict[str, Any], metadata_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    product_id = product_record.get("id")
    if product_id is not None:
        match = metadata_lookup.get(f"id:{str(product_id)}")
        if match is not None:
            return match

    link_key = normalize_product_url_key(product_record.get("link"))
    if link_key:
        match = metadata_lookup.get(f"url:{link_key}")
        if match is not None:
            return match

    texture_key = extract_texture_id(product_record.get("image_url"))
    if texture_key:
        match = metadata_lookup.get(f"tex:{texture_key}")
        if match is not None:
            return match

    return {}


def resolve_product_export_record(metadata_record: dict[str, Any], product_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    product_id = metadata_record.get("id")
    if product_id is not None:
        match = product_lookup.get(f"id:{str(product_id)}")
        if match is not None:
            return match

    brand_slug = metadata_record.get("brandSlug")
    product_slug = metadata_record.get("productSlug")
    if isinstance(brand_slug, str) and isinstance(product_slug, str):
        match = product_lookup.get(f"url:{brand_slug.strip().lower()}/{product_slug.strip().lower()}")
        if match is not None:
            return match

    preview_texture_key = extract_texture_id(metadata_record.get("previewPath"))
    if preview_texture_key:
        match = product_lookup.get(f"tex:{preview_texture_key}")
        if match is not None:
            return match
    return {}


def wait_for_task(host: str, api_key: str | None, task_uid: int, timeout_seconds: int = 600, poll_seconds: float = 0.5) -> None:
    start = time.time()
    endpoint = f"{host.rstrip('/')}/tasks/{task_uid}"
    while True:
        task = http_json("GET", endpoint, api_key=api_key)
        status = task.get("status")
        if status == "succeeded":
            return
        if status == "failed":
            error_block = task.get("error")
            raise RuntimeError(f"Meilisearch task {task_uid} failed: {error_block}")
        if time.time() - start > timeout_seconds:
            raise RuntimeError(f"Timed out waiting for Meilisearch task {task_uid} after {timeout_seconds}s.")
        time.sleep(poll_seconds)


def extract_task_uid(response: dict[str, Any], operation_name: str) -> int:
    task_uid = response.get("taskUid")
    if not isinstance(task_uid, int):
        raise RuntimeError(f"{operation_name} did not return a valid taskUid: {response}")
    return task_uid


def delete_index_if_exists(host: str, api_key: str | None, index_name: str) -> None:
    index_endpoint = f"{host.rstrip('/')}/indexes/{index_name}"
    try:
        http_json("GET", index_endpoint, api_key=api_key)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "http 404" in message or "index_not_found" in message:
            return
        raise

    response = http_json("DELETE", index_endpoint, api_key=api_key)
    wait_for_task(host, api_key, extract_task_uid(response, "delete-index"))
    print(f"Deleted existing index '{index_name}'.")


def ensure_index(host: str, api_key: str | None, index_name: str, primary_key: str = "id", vector_dimensions: int | None = None) -> None:
    delete_index_if_exists(host, api_key, index_name)
    endpoint = f"{host.rstrip('/')}/indexes"
    payload = {"uid": index_name, "primaryKey": primary_key}
    response = http_json("POST", endpoint, payload, api_key)
    wait_for_task(host, api_key, extract_task_uid(response, "create-index"))

    settings = {
        "searchableAttributes": [
            "name",
            "link",
            "image_url",
            "image_name",
            "source_image_name",
            "brandName",
            "productCategory",
            "materialType",
            "location",
            "application",
            "material_type",
            "color",
        ],
        "filterableAttributes": [
            "id",
            "product_id",
            "name",
            "link",
            "image_url",
            "image_name",
            "source_image_name",
            "brandName",
            "location",
            "application",
            "productCategory",
            "materialType",
            "material_type",
            "color",
            "locale",
        ],
        "sortableAttributes": ["product_id", "id"],
        "rankingRules": [
            "words",
            "typo",
            "proximity",
            "attribute",
            "sort",
            "exactness",
        ],
    }
    if vector_dimensions is not None:
        settings["embedders"] = {
            "default": {
                "source": "userProvided",
                "dimensions": vector_dimensions,
            }
        }
    response = http_json("PATCH", f"{host.rstrip('/')}/indexes/{index_name}/settings", settings, api_key)
    wait_for_task(host, api_key, extract_task_uid(response, "update-settings"))


def build_features(record: dict[str, Any]) -> list[str]:
    feature_values: list[str] = []
    for field in ["name", "link", "image_url", "image_name"]:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            feature_values.append(value.strip())
    return feature_values


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


def compute_embedding(image_path: Path, model: AutoModel, processor: AutoImageProcessor, device: str) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    try:
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                embedding = outputs.last_hidden_state[:, 0, :]
            else:
                raise RuntimeError(f"DINOv2 output on {image_path} does not expose a usable embedding.")

        embedding = embedding.detach().cpu().numpy()
        embedding = l2_normalize(np.asarray(embedding, dtype=np.float32))[0]
        return [float(value) for value in embedding.tolist()]
    finally:
        image.close()


def build_product_url(brand_slug: str | None, product_slug: str | None) -> str | None:
    brand = str(brand_slug or "").strip().strip("/")
    product = str(product_slug or "").strip().strip("/")
    if not brand or not product:
        return None
    return f"https://lightbeans.com/en/texture/{brand}/{product}"


def average_embedding(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Cannot average an empty embedding list.")
    matrix = np.asarray(vectors, dtype=np.float32)
    mean_vector = matrix.mean(axis=0)
    normalized = l2_normalize(np.asarray([mean_vector], dtype=np.float32))[0]
    return [float(value) for value in normalized.tolist()]


def build_folder_document(
    folder_id: str,
    folder_path: Path,
    view_paths: list[Path],
    view_embeddings: list[list[float]],
    product: dict[str, Any] | None,
) -> dict[str, Any]:
    if len(view_paths) != len(view_embeddings):
        raise ValueError(
            f"view_paths/view_embeddings length mismatch for {folder_path}: "
            f"{len(view_paths)} != {len(view_embeddings)}"
        )
    if not view_embeddings:
        raise ValueError(f"No embeddings generated for folder: {folder_path}")

    record = dict(product or {})
    product_id = record.get("id") or record.get("image_name") or record.get("link") or folder_id
    product_id = sanitize_meilisearch_id(product_id)
    representative_embedding = average_embedding(view_embeddings)
    source_image_name = record.get("image_name") or folder_path.name
    link = record.get("link") or build_product_url(record.get("brandSlug"), record.get("productSlug"))
    image_url = record.get("image_url")
    if not image_url and isinstance(record.get("previewPath"), str):
        preview_path = normalize_preview_path(record.get("previewPath"))
        if preview_path:
            image_url = f"https://cdn.lightbeans.com/textures/{preview_path.lstrip('/')}"
    record["link"] = link
    record["image_url"] = image_url or ""

    return {
        "id": folder_id,
        "folder_name": folder_path.name,
        "folder_path": folder_path.as_posix(),
        "product_id": product_id,
        "name": record.get("name") or record.get("productName"),
        "link": link,
        "image_url": image_url or "",
        "source_image_name": source_image_name,
        "image_names": [path.name for path in view_paths],
        "image_paths": [path.as_posix() for path in view_paths],
        "view_count": len(view_paths),
        "features": build_features(record),
        "brandName": record.get("brandName") or record.get("brand_name"),
        "location": as_list(record.get("location")),
        "application": as_list(record.get("application")),
        "productCategory": as_list(record.get("productCategory") or record.get("product_category")),
        "materialType": as_list(record.get("materialType") or record.get("material_type")),
        "product_category": as_list(record.get("productCategory") or record.get("product_category")),
        "material_type": as_list(record.get("materialType") or record.get("material_type")),
        "color": as_list(record.get("color")),
        "locale": record.get("locale"),
        "embedding_vectors": view_embeddings,
        "_vectors": {"default": representative_embedding},
    }


def upload_documents(host: str, api_key: str | None, index_name: str, documents: list[dict[str, Any]]) -> None:
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/documents"
    response = http_json("POST", endpoint, documents, api_key)
    task_uid = extract_task_uid(response, "upload-documents")
    wait_for_task(host, api_key, task_uid)


def estimate_json_bytes(payload: Any) -> int:
    return len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def upload_documents_with_payload_limit(
    host: str,
    api_key: str | None,
    index_name: str,
    documents: list[dict[str, Any]],
    max_payload_bytes: int,
) -> list[int]:
    if max_payload_bytes <= 0:
        raise ValueError(f"max_payload_bytes must be > 0, got {max_payload_bytes}.")
    if not documents:
        return []

    chunk_sizes: list[int] = []
    current_chunk: list[dict[str, Any]] = []
    current_bytes = 2  # []

    for doc in documents:
        doc_bytes = estimate_json_bytes(doc)
        min_doc_container_bytes = doc_bytes + 2
        if min_doc_container_bytes > max_payload_bytes:
            doc_id = doc.get("id")
            raise RuntimeError(
                f"Single document exceeds payload limit ({doc_bytes} bytes > {max_payload_bytes} bytes). "
                f"Document id: {doc_id!r}. Reduce per-document payload before uploading."
            )

        add_bytes = doc_bytes + (1 if current_chunk else 0)  # comma between array items
        if current_chunk and current_bytes + add_bytes > max_payload_bytes:
            upload_documents(host, api_key, index_name, current_chunk)
            chunk_sizes.append(len(current_chunk))
            current_chunk = []
            current_bytes = 2

        current_chunk.append(doc)
        current_bytes += doc_bytes + (1 if len(current_chunk) > 1 else 0)

    if current_chunk:
        upload_documents(host, api_key, index_name, current_chunk)
        chunk_sizes.append(len(current_chunk))

    return chunk_sizes


def search_by_image(
    host: str,
    api_key: str | None,
    index_name: str,
    image_path: Path,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
    limit: int = 10,
):
    vector = compute_embedding(image_path, model, processor, device)
    payload = {
        "vector": vector,
        "limit": limit,
        "attributesToRetrieve": [
            "product_id",
            "name",
            "link",
            "image_url",
            "image_name",
            "image_path",
        ],
        "hybrid": {"semanticRatio": 1.0, "embedder": "default"},
    }
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/search"
    return http_json("POST", endpoint, payload, api_key)


def search_meilisearch(host: str, api_key: str | None, index_name: str, vector: list[float], limit: int = 10) -> dict[str, Any]:
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/search"
    payload = {
        "vector": vector,
        "limit": limit,
        "attributesToRetrieve": [
            "product_id",
            "name",
            "link",
            "image_url",
            "image_name",
            "image_path",
            "preview_path",
            "product_url",
            "brand_slug",
            "product_slug",
        ],
        "hybrid": {"semanticRatio": 1.0, "embedder": "default"},
        "showRankingScore": True,
    }
    return http_json("POST", endpoint, payload, api_key)


def format_match_for_viewer(hit: dict[str, Any], rank: int) -> dict[str, Any]:
    preview_path = hit.get("preview_path") or hit.get("previewPath")
    image_url = hit.get("image_url") or hit.get("imageUrl") or preview_path
    product_url = hit.get("link") or hit.get("product_url") or hit.get("productUrl")
    if not product_url:
        brand_slug = hit.get("brand_slug") or hit.get("brandSlug")
        product_slug = hit.get("product_slug") or hit.get("productSlug")
        product_url = build_product_url(brand_slug, product_slug)
    score = (
        hit.get("score")
        if hit.get("score") is not None
        else hit.get("rankingScore")
        if hit.get("rankingScore") is not None
        else hit.get("_rankingScore")
        if hit.get("_rankingScore") is not None
        else hit.get("similarity")
        if hit.get("similarity") is not None
        else hit.get("distance")
    )
    return {
        "rank": rank,
        "product_id": hit.get("product_id") or hit.get("id"),
        "name": hit.get("name") or hit.get("product_name") or hit.get("productName"),
        "link": product_url,
        "image_url": image_url,
        "image_name": hit.get("image_name") or hit.get("imageName"),
        "product_url": product_url,
        "preview_path": preview_path,
        "texture_url": image_url,
        "score": score,
        "ranking_score": hit.get("rankingScore") or hit.get("_rankingScore"),
    }


def search_by_image_for_viewer(
    host: str,
    api_key: str | None,
    index_name: str,
    image_path: Path,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
    limit: int = 10,
) -> dict[str, Any]:
    vector = compute_embedding(image_path, model, processor, device)
    result = search_meilisearch(host, api_key, index_name, vector, limit=limit)
    hits = result.get("hits", []) if isinstance(result, dict) else []
    matches = [format_match_for_viewer(hit, idx) for idx, hit in enumerate(hits[:limit], start=1)]
    return {
        "image": str(image_path),
        "embedding_dim": len(vector),
        "matches": matches,
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    env_values = load_env(project_root / ".env")
    parser = argparse.ArgumentParser(description="Index DINOv2 texture embeddings into Meilisearch.")
    parser.add_argument(
        "--product-export",
        type=Path,
        default=Path(DEFAULT_PRODUCT_EXPORT),
        help="Optional product export JSON used to enrich image_name/link/image_url matching.",
    )
    parser.add_argument(
        "--filter-metadata-export",
        type=Path,
        default=Path(DEFAULT_FILTER_METADATA_EXPORT),
        help="Product metadata JSON used as the primary source for indexed documents.",
    )
    parser.add_argument("--textures-dir", type=Path, default=Path(DEFAULT_TEXTURE_DIR), help="Directory containing texture images to index (e.g. generated views).")
    parser.add_argument("--host", default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700", help="Meilisearch host.")
    parser.add_argument("--key", default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"), help="Meilisearch API key.")
    parser.add_argument("--index", default=env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or MEILI_INDEX, help="Target Meilisearch index name.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N products (0 = all).")
    parser.add_argument("--upload-batch-size", type=int, default=256, help="Number of documents uploaded per Meilisearch request.")
    parser.add_argument("--trace", dest="trace", action="store_true", help="Print progression traces while indexing.")
    parser.add_argument("--no-trace", dest="trace", action="store_false", help="Disable progression traces while indexing.")
    parser.set_defaults(trace=True)
    parser.add_argument("--trace-every", type=int, default=100, help="Emit progress every N metadata records when tracing.")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"], help="Device for DINOv2 inference.")
    parser.add_argument("--query-image", type=Path, help="Optional image path used to run a vector search after indexing.")
    parser.add_argument("--search-limit", type=int, default=10, help="Number of hits to return from the optional vector search.")
    parser.add_argument("--search-output", type=Path, help="Optional output JSON path for viewer-compatible search payload.")
    parser.add_argument(
        "--expected-views-per-folder",
        type=int,
        default=48,
        help="Expected number of view images per folder (excluding contact_sheet).",
    )
    parser.add_argument(
        "--max-upload-payload-mib",
        type=int,
        default=DEFAULT_MAX_UPLOAD_PAYLOAD_MIB,
        help="Maximum JSON payload size (MiB) per Meilisearch document upload request.",
    )
    args = parser.parse_args()
    if env_values.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = env_values["HF_TOKEN"]
    if os.getenv("HF_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")
    return args


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")
    return device_arg


def main() -> int:
    args = parse_args()
    if not args.filter_metadata_export.exists():
        raise FileNotFoundError(f"Filter metadata export not found: {args.filter_metadata_export}")

    if not args.textures_dir.exists():
        raise FileNotFoundError(f"Textures directory not found: {args.textures_dir}")

    device = choose_device(args.device)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME, token=hf_token)
    image_model = AutoModel.from_pretrained(MODEL_NAME, token=hf_token)
    image_model.to(device)
    image_model.eval()

    metadata_records = load_filter_metadata_export(args.filter_metadata_export)
    if args.limit > 0:
        metadata_records = metadata_records[: args.limit]
    total_records = len(metadata_records)
    metadata_lookup = build_filter_metadata_lookup(metadata_records)

    product_records: list[dict[str, Any]] = []
    if args.product_export.exists():
        product_records = load_product_export(args.product_export)
    product_lookup = build_product_lookup(product_records)

    if args.upload_batch_size <= 0:
        raise ValueError(f"--upload-batch-size must be > 0, got {args.upload_batch_size}.")
    if args.trace_every <= 0:
        raise ValueError(f"--trace-every must be > 0, got {args.trace_every}.")
    if args.expected_views_per_folder <= 0:
        raise ValueError(
            f"--expected-views-per-folder must be > 0, got {args.expected_views_per_folder}."
        )
    if args.max_upload_payload_mib <= 0:
        raise ValueError(
            f"--max-upload-payload-mib must be > 0, got {args.max_upload_payload_mib}."
        )
    max_upload_payload_bytes = int(args.max_upload_payload_mib * 1024 * 1024)

    upload_buffer: list[dict[str, Any]] = []
    indexed_folders = 0
    total_indexed_views = 0
    skipped_with_invalid_view_count = 0
    skipped_without_product_match = 0
    skipped_without_folder_match = 0
    skipped_duplicate_folder_match = 0
    view_folders = collect_view_folders(args.textures_dir)
    if not view_folders:
        raise RuntimeError(f"No supported images found in {args.textures_dir} (excluding contact_sheet files).")
    total_folders = len(view_folders)
    folder_lookup = build_view_folder_lookup(view_folders)

    vector_dimensions: int | None = None
    index_initialized = False
    processed_folder_ids: set[str] = set()
    if args.trace:
        print(
            "[trace] Starting DINOv2 indexing "
            f"(metadata={total_records}, product_export={len(product_records)}, texture_folders={total_folders})."
        )

    for record_index, metadata_record in enumerate(metadata_records, start=1):
        product_record = resolve_product_export_record(metadata_record, product_lookup)
        if product_record is None or not product_record:
            skipped_without_product_match += 1
            if args.trace:
                print(f"[trace] Skipped {record_index}/{total_records}: no matching product export entry for metadata id '{metadata_record.get('id')}'.")
            continue
        metadata_hit = resolve_filter_metadata(product_record, metadata_lookup)
        resolved_record = dict(metadata_record)
        resolved_record.update(metadata_hit)
        resolved_record.update(product_record)
        resolved = resolve_view_folder_for_record(resolved_record, folder_lookup)
        if resolved is None:
            skipped_without_folder_match += 1
            if args.trace:
                print(f"[trace] Skipped {record_index}/{total_records}: no matching view folder for metadata id '{metadata_record.get('id')}'.")
            continue
        folder_path, view_paths = resolved

        if len(view_paths) != args.expected_views_per_folder:
            skipped_with_invalid_view_count += 1
            if args.trace:
                print(
                    f"[trace] Skipped {record_index}/{total_records}: folder '{folder_path.name}' "
                    f"expected {args.expected_views_per_folder} views but found {len(view_paths)}."
                )
            continue

        folder_id = sanitize_meilisearch_id(folder_path.name)
        if folder_id in processed_folder_ids:
            skipped_duplicate_folder_match += 1
            if args.trace:
                print(f"[trace] Skipped {record_index}/{total_records}: duplicate folder match '{folder_path.name}'.")
            continue
        processed_folder_ids.add(folder_id)

        if "id" not in resolved_record:
            fallback_id = (
                resolved_record.get("id")
                or resolved_record.get("image_name")
                or resolved_record.get("link")
                or resolved_record.get("image_url")
                or resolved_record.get("previewPath")
            )
            if fallback_id:
                resolved_record["id"] = sanitize_meilisearch_id(fallback_id)

        with ThreadPoolExecutor(max_workers=len(view_paths)) as folder_executor:
            view_embeddings = list(
                folder_executor.map(
                    lambda texture_path: compute_embedding(texture_path, image_model, image_processor, device),
                    view_paths,
                )
            )
        doc = build_folder_document(
            folder_id=folder_id,
            folder_path=folder_path,
            view_paths=view_paths,
            view_embeddings=view_embeddings,
            product=resolved_record,
        )
        if vector_dimensions is None:
            vector_dimensions = len(doc["embedding"])
        if not index_initialized:
            ensure_index(args.host, args.key, args.index, vector_dimensions=vector_dimensions)
            index_initialized = True

        upload_buffer.append(doc)
        indexed_folders += 1
        total_indexed_views += len(view_paths)
        if args.trace and (record_index == 1 or record_index % args.trace_every == 0 or record_index == total_records):
            print(
                f"[trace] Progress {record_index}/{total_records}: indexed={indexed_folders}, "
                f"skipped_no_product={skipped_without_product_match}, "
                f"skipped_no_folder={skipped_without_folder_match}, "
                f"skipped_invalid_views={skipped_with_invalid_view_count}, "
                f"skipped_duplicates={skipped_duplicate_folder_match}."
            )
        if len(upload_buffer) >= args.upload_batch_size:
            upload_documents_with_payload_limit(
                args.host,
                args.key,
                args.index,
                upload_buffer,
                max_payload_bytes=max_upload_payload_bytes,
            )
            upload_buffer.clear()
            if args.trace:
                print(f"[trace] Uploaded {indexed_folders} / {total_records} folder documents.")

    if indexed_folders == 0:
        raise RuntimeError(
            f"No indexable view folders found. "
            f"Checked {total_records} metadata records in {args.filter_metadata_export} "
            f"with expected view count {args.expected_views_per_folder}."
        )

    if upload_buffer:
        upload_documents_with_payload_limit(
            args.host,
            args.key,
            args.index,
            upload_buffer,
            max_payload_bytes=max_upload_payload_bytes,
        )
        upload_buffer.clear()

    print(
        f"Indexed {indexed_folders} texture folders ({total_indexed_views} views) "
        f"in Meilisearch index '{args.index}'."
    )
    print(f"Skipped {skipped_with_invalid_view_count} folders with invalid view count.")
    print(f"Skipped {skipped_without_product_match} metadata records without matching product export entry.")
    print(f"Skipped {skipped_without_folder_match} metadata records without matching view folder.")
    print(f"Skipped {skipped_duplicate_folder_match} duplicate metadata matches for already indexed folders.")

    if args.query_image:
        if not args.query_image.exists():
            raise FileNotFoundError(f"Query image not found: {args.query_image}")
        search_payload = search_by_image_for_viewer(
            args.host,
            args.key,
            args.index,
            args.query_image,
            image_model,
            image_processor,
            device,
            limit=args.search_limit,
        )
        rendered = json.dumps(search_payload, ensure_ascii=False, indent=2)
        if args.search_output:
            args.search_output.parent.mkdir(parents=True, exist_ok=True)
            args.search_output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered[:4000])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
