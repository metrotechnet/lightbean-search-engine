#!/usr/bin/env python3
"""Match a masked image region against the DINOv2 texture embedding index in Meilisearch.

Workflow:
1. Load an input image and a binary mask.
2. Extract the rectangular bounding box around the mask.
3. Crop the image to that box and compute a DINOv2 embedding.
4. Search the Meilisearch texture index for the 5 most similar products.
5. Return the matches with their scores and texture URLs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import urljoin

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-small"
DEFAULT_INDEX = "lightbeans-texture-dinov2"
TEXTURE_CDN = "https://cdn.lightbeans.com/textures/"
PRODUCT_URL_BASE = "https://lightbeans.com/en/texture/"


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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    env_values = load_env(project_root / ".env")
    parser = argparse.ArgumentParser(description="Crop a masked region from an image, embed it with DINOv2, and match it against a Meilisearch texture index.")
    parser.add_argument("--image", type=Path, required=True, help="Input image file.")
    parser.add_argument("--mask", type=Path, required=True, help="Binary mask image, same size as the input image.")
    parser.add_argument("--margin", type=int, default=10, help="Pixel padding added around the mask bounding box.")
    parser.add_argument("--host", default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700", help="Meilisearch host.")
    parser.add_argument("--key", default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"), help="Meilisearch API key.")
    parser.add_argument("--index", default=env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or DEFAULT_INDEX, help="Meilisearch index to search.")
    parser.add_argument("--limit", type=int, default=10, help="Number of matches to return.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for DINOv2 inference.")
    parser.add_argument("--output", type=Path, help="Optional JSON file to save the result.")
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


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[:, :, 0]
        else:
            mask = mask.any(axis=2)
    mask = mask.astype(np.uint8)
    return mask > 0


def extract_bounding_box(mask: np.ndarray, margin: int = 0) -> tuple[int, int, int, int]:
    binary = normalize_mask(mask)
    if not np.any(binary):
        raise ValueError("Mask is empty; no non-zero pixels were found.")

    ys, xs = np.nonzero(binary)
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(mask.shape[1], int(xs.max()) + 1 + margin)
    y2 = min(mask.shape[0], int(ys.max()) + 1 + margin)
    return x1, y1, x2, y2


def crop_image_from_mask(image: np.ndarray, mask: np.ndarray, margin: int = 10) -> tuple[np.ndarray, dict[str, int]]:
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image shape {image.shape[:2]} and mask shape {mask.shape[:2]} do not match.")
    x1, y1, x2, y2 = extract_bounding_box(mask, margin=margin)
    cropped = image[y1:y2, x1:x2]
    bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2 - x1, "height": y2 - y1}
    return cropped, bbox


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vector, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vector / norms


def compute_embedding(image: Image.Image, model: AutoModel, processor: AutoImageProcessor, device: str) -> list[float]:
    pixel_values = processor(images=image, return_tensors="pt")
    pixel_values = {key: value.to(device) for key, value in pixel_values.items()}
    with torch.no_grad():
        outputs = model(**pixel_values)
        if hasattr(outputs, "last_hidden_state"):
            embedding = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embedding = outputs.pooler_output
        else:
            raise RuntimeError("DINOv2 output does not expose a usable embedding.")

    embedding = l2_normalize(embedding.detach().cpu().numpy().astype(np.float32))[0]
    return [float(value) for value in embedding.tolist()]


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


def configure_index_embedder(host: str, api_key: str | None, index_name: str, vector_dimensions: int) -> None:
    settings = {
        "embedders": {
            "default": {
                "source": "userProvided",
                "dimensions": vector_dimensions,
            }
        }
    }
    response = http_json("PATCH", f"{host.rstrip('/')}/indexes/{index_name}/settings", settings, api_key)
    wait_for_task(host, api_key, extract_task_uid(response, "update-settings"))


def search_meilisearch(host: str, api_key: str | None, index_name: str, vector: list[float], limit: int = 10) -> dict[str, Any]:
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/search"
    payload = {
        "vector": vector,
        "limit": limit,
        "attributesToRetrieve": [
            "product_id",
            "product_name",
            "brand_name",
            "brand_slug",
            "product_slug",
            "product_category",
            "material_type",
            "location",
            "application",
            "color",
            "preview_path",
            "image_path",
        ],
        "hybrid": {"semanticRatio": 1.0, "embedder": "default"},
        "showRankingScore": True,
    }
    try:
        return http_json("POST", endpoint, payload, api_key)
    except RuntimeError as exc:
        message = str(exc)
        if "Cannot find embedder with name `default`" not in message and "invalid_search_embedder" not in message:
            raise
        configure_index_embedder(host, api_key, index_name, len(vector))
        return http_json("POST", endpoint, payload, api_key)


def build_texture_url(preview_path: str | None) -> str | None:
    if not preview_path:
        return None
    normalized = str(preview_path).strip()
    if not normalized:
        return None
    if normalized.startswith("http://") or normalized.startswith("https://"):
        return normalized
    return urljoin(TEXTURE_CDN, normalized.lstrip("/"))


def build_product_url(brand_slug: str | None, product_slug: str | None) -> str | None:
    brand = str(brand_slug or "").strip().strip("/")
    product = str(product_slug or "").strip().strip("/")
    if not brand or not product:
        return None
    return f"{PRODUCT_URL_BASE.rstrip('/')}/{brand}/{product}"


def format_match(hit: dict[str, Any], rank: int) -> dict[str, Any]:
    preview_path = hit.get("preview_path") or hit.get("previewPath")
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
        "product_name": hit.get("product_name") or hit.get("productName"),
        "brand_name": hit.get("brand_name") or hit.get("brandName"),
        "product_category": hit.get("product_category") or hit.get("productCategory"),
        "material_type": hit.get("material_type") or hit.get("materialType"),
        "location": hit.get("location"),
        "application": hit.get("application"),
        "color": hit.get("color"),
        "brand_slug": brand_slug,
        "product_slug": product_slug,
        "product_url": product_url,
        "preview_path": preview_path,
        "texture_url": build_texture_url(preview_path),
        "score": score,
        "ranking_score": hit.get("rankingScore") or hit.get("_rankingScore"),
    }


def main() -> int:
    args = parse_args()
    image_path = args.image
    mask_path = args.mask

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    image = Image.open(image_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("L")

    try:
        image_array = np.asarray(image)
        mask_array = np.asarray(mask_image)
        cropped, bbox = crop_image_from_mask(image_array, mask_array, margin=args.margin)
        cropped_image = Image.fromarray(cropped.astype(np.uint8))

        device = choose_device(args.device)
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME, token=hf_token)
        model = AutoModel.from_pretrained(MODEL_NAME, token=hf_token)
        model.to(device)
        model.eval()

        embedding = compute_embedding(cropped_image, model, image_processor, device)
        result = search_meilisearch(args.host, args.key, args.index, embedding, limit=args.limit)
        hits = result.get("hits", []) if isinstance(result, dict) else []
        matches = [format_match(hit, index) for index, hit in enumerate(hits[: args.limit], start=1)]

        summary = {
            "image": str(image_path),
            "mask": str(mask_path),
            "bbox": bbox,
            "embedding_dim": len(embedding),
            "matches": matches,
        }

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        image.close()
        mask_image.close()


if __name__ == "__main__":
    raise SystemExit(main())
