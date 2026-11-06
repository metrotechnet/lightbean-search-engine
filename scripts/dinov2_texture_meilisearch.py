#!/usr/bin/env python3
"""Index texture-image embeddings from DINOv2 into Meilisearch.

This script loads products from the Lightbeans export, resolves each product's downloaded texture file,
computes a DINOv2 image embedding, and uploads the product metadata + embedding to Meilisearch.
The resulting index can then be queried by text or by nearest-neighbor similarity using the stored vector.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-small"
DEFAULT_PRODUCT_EXPORT = "data/platform-products-production-v2-en.json"
DEFAULT_TEXTURE_DIR = "texture"
DEFAULT_MAPPING_PATH = "texture/texture_product_mapping.json"
DEFAULT_INDEX = "lightbeans-texture-dinov2"


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
    data = None if payload is None else json.dumps(payload).encode("utf-8")
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


def ensure_index(host: str, api_key: str | None, index_name: str, primary_key: str = "product_id") -> None:
    endpoint = f"{host.rstrip('/')}/indexes"
    payload = {"uid": index_name, "primaryKey": primary_key}
    try:
        http_json("POST", endpoint, payload, api_key)
    except RuntimeError as exc:
        if "already exists" not in str(exc).lower() and "index_already_exists" not in str(exc).lower():
            raise

    settings = {
        "searchableAttributes": [
            "product_name",
            "brand_name",
            "product_category",
            "material_type",
            "location",
            "application",
            "color",
        ],
        "filterableAttributes": [
            "product_id",
            "brand_name",
            "product_category",
            "material_type",
            "location",
            "application",
            "color",
            "locale",
        ],
        "sortableAttributes": ["product_id"],
        "rankingRules": [
            "words",
            "typo",
            "proximity",
            "attribute",
            "sort",
            "exactness",
        ],
    }
    http_json("PATCH", f"{host.rstrip('/')}/indexes/{index_name}/settings", settings, api_key)


def build_features(record: dict[str, Any]) -> list[str]:
    feature_values: list[str] = []
    for field in [
        "productName",
        "brandName",
        "productCategory",
        "materialType",
        "location",
        "application",
        "color",
        "locale",
    ]:
        for value in as_list(record.get(field)):
            feature_values.append(value)
    return feature_values


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


def build_document(product: dict[str, Any], image_path: Path, embedding: list[float]) -> dict[str, Any]:
    product_id = str(product.get("id"))
    doc = {
        "product_id": product_id,
        "id": product_id,
        "product_name": product.get("productName"),
        "brand_name": product.get("brandName"),
        "product_category": as_list(product.get("productCategory")),
        "material_type": as_list(product.get("materialType")),
        "location": as_list(product.get("location")),
        "application": as_list(product.get("application")),
        "color": as_list(product.get("color")),
        "locale": product.get("locale"),
        "preview_path": product.get("previewPath"),
        "image_path": image_path.as_posix(),
        "features": build_features(product),
        "embedding": embedding,
        "_vectors": {"default": embedding},
    }
    return doc


def upload_documents(host: str, api_key: str | None, index_name: str, documents: list[dict[str, Any]]) -> None:
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/documents"
    http_json("POST", endpoint, documents, api_key)


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
            "product_name",
            "brand_name",
            "product_category",
            "material_type",
            "location",
            "application",
            "color",
            "image_path",
        ],
        "hybrid": {"semanticRatio": 1.0, "embedder": "default"},
    }
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/search"
    return http_json("POST", endpoint, payload, api_key)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    env_values = load_env(project_root / ".env")
    parser = argparse.ArgumentParser(description="Index DINOv2 texture embeddings into Meilisearch.")
    parser.add_argument("--product-export", type=Path, default=Path(DEFAULT_PRODUCT_EXPORT), help="Product export JSON file.")
    parser.add_argument("--textures-dir", type=Path, default=Path(DEFAULT_TEXTURE_DIR), help="Directory containing downloaded texture files.")
    parser.add_argument("--mapping-json", type=Path, default=Path(DEFAULT_MAPPING_PATH), help="Texture-to-product mapping JSON generated by download_texture_images.py.")
    parser.add_argument("--host", default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700", help="Meilisearch host.")
    parser.add_argument("--key", default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"), help="Meilisearch API key.")
    parser.add_argument("--index", default=env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or DEFAULT_INDEX, help="Target Meilisearch index name.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N products (0 = all).")
    parser.add_argument("--batch-size", type=int, default=8, help="Image batch size for DINOv2 inference.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for DINOv2 inference.")
    parser.add_argument("--query-image", type=Path, help="Optional image path used to run a vector search after indexing.")
    parser.add_argument("--search-limit", type=int, default=10, help="Number of hits to return from the optional vector search.")
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
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME, token=hf_token)
    image_model = AutoModel.from_pretrained(MODEL_NAME, token=hf_token)
    image_model.to(device)
    image_model.eval()

    products = load_product_export(product_export)
    if args.limit > 0:
        products = products[: args.limit]

    documents: list[dict[str, Any]] = []
    skipped = 0
    for record in products:
        product_id = record.get("id")
        if product_id is None:
            continue

        texture_path = resolve_texture_path(record, args.textures_dir, args.mapping_json)
        if texture_path is None or not texture_path.exists():
            skipped += 1
            continue

        embedding = compute_embedding(texture_path, image_model, image_processor, device)
        documents.append(build_document(record, texture_path, embedding))

    ensure_index(args.host, args.key, args.index)
    upload_documents(args.host, args.key, args.index, documents)
    print(f"Indexed {len(documents)} product texture embeddings in Meilisearch index '{args.index}'.")
    print(f"Skipped {skipped} products without a usable texture file.")

    if args.query_image:
        if not args.query_image.exists():
            raise FileNotFoundError(f"Query image not found: {args.query_image}")
        result = search_by_image(
            args.host,
            args.key,
            args.index,
            args.query_image,
            image_model,
            image_processor,
            device,
            limit=args.search_limit,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
