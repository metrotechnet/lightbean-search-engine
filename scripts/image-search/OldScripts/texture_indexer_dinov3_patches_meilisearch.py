#!/usr/bin/env python3
"""Index texture-image embeddings from DINOv3 patch tokens into Meilisearch.

This script mirrors the DINOv2 indexing flow but uses DINOv3 and builds vectors
from patch tokens (mean/max pooling over patch embeddings).
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
from pathlib import Path
from typing import Any
from urllib import error, request

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_PRODUCT_EXPORT = "texture/lightbeans_main_page_textures.json"
DEFAULT_TEXTURE_DIR = "texture/main_page"
MEILI_INDEX = "lightbeans-texture2-dinov3-patches"
DEFAULT_INDEX = MEILI_INDEX


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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_product_export(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [item for item in data if isinstance(item, dict)]


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


def ensure_index(host: str, api_key: str | None, index_name: str, primary_key: str = "product_id", vector_dimensions: int | None = None) -> None:
    endpoint = f"{host.rstrip('/')}/indexes"
    payload = {"uid": index_name, "primaryKey": primary_key}
    try:
        response = http_json("POST", endpoint, payload, api_key)
        wait_for_task(host, api_key, extract_task_uid(response, "create-index"))
    except RuntimeError as exc:
        if "already exists" not in str(exc).lower() and "index_already_exists" not in str(exc).lower():
            raise

    settings = {
        "searchableAttributes": ["name", "link", "image_url", "image_name"],
        "filterableAttributes": ["product_id", "name", "link", "image_url", "image_name"],
        "sortableAttributes": ["product_id"],
        "rankingRules": ["words", "typo", "proximity", "attribute", "sort", "exactness"],
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


def patch_pool(last_hidden_state: torch.Tensor, mode: str) -> torch.Tensor:
    if last_hidden_state.ndim != 3:
        raise RuntimeError(f"Unexpected hidden-state shape: {tuple(last_hidden_state.shape)}")

    if mode == "cls":
        return last_hidden_state[:, 0, :]

    tokens = last_hidden_state[:, 1:, :] if last_hidden_state.shape[1] > 1 else last_hidden_state
    if tokens.shape[1] == 0:
        tokens = last_hidden_state

    if mode == "mean":
        return tokens.mean(dim=1)
    if mode == "max":
        return tokens.max(dim=1).values
    raise ValueError(f"Unknown patch pooling mode '{mode}'.")


def compute_embedding(image_path: Path, model: AutoModel, processor: AutoImageProcessor, device: str, patch_pooling: str) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    try:
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

            if hasattr(outputs, "last_hidden_state"):
                embedding_tensor = patch_pool(outputs.last_hidden_state, patch_pooling)
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embedding_tensor = outputs.pooler_output
            else:
                raise RuntimeError(f"Model output on {image_path} does not expose usable patch embeddings.")

        embedding = embedding_tensor.detach().cpu().numpy()
        embedding = l2_normalize(np.asarray(embedding, dtype=np.float32))[0]
        return [float(value) for value in embedding.tolist()]
    finally:
        image.close()


def build_document(product: dict[str, Any], image_path: Path, embedding: list[float]) -> dict[str, Any]:
    raw_product_id = product.get("id") or product.get("image_name") or product.get("link") or image_path.stem
    product_id = sanitize_meilisearch_id(raw_product_id)
    return {
        "product_id": product_id,
        "id": product_id,
        "name": product.get("name"),
        "link": product.get("link"),
        "image_url": product.get("image_url") or "",
        "image_name": product.get("image_name") or image_path.name,
        "image_path": image_path.as_posix(),
        "features": build_features(product),
        "embedding": embedding,
        "_vectors": {"default": embedding},
    }


def upload_documents(
    host: str,
    api_key: str | None,
    index_name: str,
    documents: list[dict[str, Any]],
    batch_size: int = 100,
) -> None:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/documents"
    total = len(documents)
    if total == 0:
        return
    for start in range(0, total, batch_size):
        batch = documents[start : start + batch_size]
        response = http_json("POST", endpoint, batch, api_key, retries=5, retry_delay_seconds=2.0)
        task_uid = extract_task_uid(response, "upload-documents")
        wait_for_task(host, api_key, task_uid)


def search_by_image(
    host: str,
    api_key: str | None,
    index_name: str,
    image_path: Path,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
    patch_pooling: str,
    limit: int = 10,
):
    vector = compute_embedding(image_path, model, processor, device, patch_pooling)
    payload = {
        "vector": vector,
        "limit": limit,
        "attributesToRetrieve": ["product_id", "name", "link", "image_url", "image_name", "image_path"],
        "hybrid": {"semanticRatio": 1.0, "embedder": "default"},
    }
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/search"
    return http_json("POST", endpoint, payload, api_key)


def build_product_url(brand_slug: str | None, product_slug: str | None) -> str | None:
    brand = str(brand_slug or "").strip().strip("/")
    product = str(product_slug or "").strip().strip("/")
    if not brand or not product:
        return None
    return f"https://lightbeans.com/en/texture/{brand}/{product}"


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
    patch_pooling: str,
    limit: int = 10,
) -> dict[str, Any]:
    vector = compute_embedding(image_path, model, processor, device, patch_pooling)
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
    parser = argparse.ArgumentParser(description="Index DINOv3 patch embeddings into Meilisearch.")
    parser.add_argument("--product-export", type=Path, default=Path(DEFAULT_PRODUCT_EXPORT), help="Product export JSON file.")
    parser.add_argument("--textures-dir", type=Path, default=Path(DEFAULT_TEXTURE_DIR), help="Directory containing downloaded texture files.")
    parser.add_argument("--mapping-json", type=Path, default=None, help="Optional file-to-product mapping JSON for legacy texture exports.")
    parser.add_argument("--host", default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700", help="Meilisearch host.")
    parser.add_argument("--key", default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"), help="Meilisearch API key.")
    parser.add_argument("--index", default=env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or MEILI_INDEX, help="Target Meilisearch index name.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="DINOv3 model name on Hugging Face.")
    parser.add_argument("--patch-pool", default="mean", choices=["mean", "max", "cls"], help="How to pool patch tokens into one vector.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N products (0 = all).")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of documents per Meilisearch upload batch.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for DINOv3 inference.")
    parser.add_argument("--query-image", type=Path, help="Optional image path used to run a vector search after indexing.")
    parser.add_argument("--search-limit", type=int, default=10, help="Number of hits to return from the optional vector search.")
    parser.add_argument("--search-output", type=Path, help="Optional output JSON path for viewer-compatible search payload.")
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
    product_export = args.product_export
    if not product_export.exists():
        raise FileNotFoundError(f"Product export not found: {product_export}")
    if not args.textures_dir.exists():
        raise FileNotFoundError(f"Textures directory not found: {args.textures_dir}")

    device = choose_device(args.device)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    image_processor = AutoImageProcessor.from_pretrained(args.model_name, token=hf_token)
    image_model = AutoModel.from_pretrained(args.model_name, token=hf_token)
    image_model.to(device)
    image_model.eval()

    products = load_product_export(product_export)
    if args.limit > 0:
        products = products[: args.limit]

    documents: list[dict[str, Any]] = []
    skipped = 0
    for record in products:
        product_id = record.get("id") or record.get("image_name") or record.get("link") or record.get("image_url")
        if product_id is None:
            continue
        if "id" not in record and product_id is not None:
            record["id"] = sanitize_meilisearch_id(product_id)

        texture_path = resolve_texture_path(record, args.textures_dir, args.mapping_json)
        if texture_path is None or not texture_path.exists():
            skipped += 1
            continue

        embedding = compute_embedding(texture_path, image_model, image_processor, device, args.patch_pool)
        documents.append(build_document(record, texture_path, embedding))

    vector_dimensions = len(documents[0]["embedding"]) if documents else None
    ensure_index(args.host, args.key, args.index, vector_dimensions=vector_dimensions)
    upload_documents(args.host, args.key, args.index, documents, batch_size=args.batch_size)
    print(f"Indexed {len(documents)} DINOv3 patch embeddings in Meilisearch index '{args.index}'.")
    print(f"Skipped {skipped} products without a usable texture file.")

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
            patch_pooling=args.patch_pool,
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
