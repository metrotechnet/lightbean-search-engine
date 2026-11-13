#!/usr/bin/env python3
"""Match a masked image region against texture embedding indexes in Meilisearch.

Workflow:
1. Load an input image and a binary mask.
2. Find the largest axis-aligned rectangle fully contained in the mask.
3. Crop the image to that box and run HSI prefilter + DINOv2 reranking.
4. Search the Meilisearch texture index for the 5 most similar products.
5. Return the matches with their scores and texture URLs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import urljoin, urlparse

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME_DINOV2 = "facebook/dinov2-small"
DEFAULT_HSI_DINOV2_INDEX = "lightbeans-texture-hsi-dinov2-views"
TEXTURE_CDN = "https://cdn.lightbeans.com/textures/"
PRODUCT_URL_BASE = "https://lightbeans.com/en/texture/"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEARCH_ATTRIBUTES_TO_RETRIEVE = [
    "id",
    "brandName",
    "productName",
    "link",
    "image_url",
    "dinov2_embedding_vectors",
]


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


def format_payload_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < (1024 * 1024):
        return f"{size_bytes / 1024.0:.1f} KiB"
    return f"{size_bytes / (1024.0 * 1024.0):.2f} MiB"


def log_server_metric(
    *,
    enabled: bool,
    step_name: str | None,
    method: str,
    url: str,
    status: int | str,
    request_bytes: int,
    response_bytes: int,
    elapsed_seconds: float,
) -> None:
    # if not enabled:
    #     return
    path = urlparse(url).path or url
    label = step_name or "request"
    sys.stderr.write(
        f"[meili] {label} {method} {path} status={status} "
        f"req={format_payload_size(request_bytes)} resp={format_payload_size(response_bytes)} "
        f"time={elapsed_seconds:.3f}s\n"
    )
    sys.stderr.flush()


def http_json_with_metrics(
    method: str,
    url: str,
    payload: dict | list | None = None,
    api_key: str | None = None,
    *,
    log_metrics: bool = False,
    step_name: str | None = None,
):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_bytes = len(data) if data is not None else 0
    req = request.Request(url, data=data, headers=headers, method=method)
    started_at = time.perf_counter()
    try:
        with request.urlopen(req, timeout=120) as resp:
            body_bytes = resp.read()
            elapsed_seconds = time.perf_counter() - started_at
            log_server_metric(
                enabled=log_metrics,
                step_name=step_name,
                method=method,
                url=url,
                status=getattr(resp, "status", "unknown"),
                request_bytes=request_bytes,
                response_bytes=len(body_bytes),
                elapsed_seconds=elapsed_seconds,
            )
            body = body_bytes.decode("utf-8")
            return json.loads(body) if body else {}
    except error.HTTPError as exc:
        body_bytes = exc.read()
        elapsed_seconds = time.perf_counter() - started_at
        log_server_metric(
            enabled=log_metrics,
            step_name=step_name,
            method=method,
            url=url,
            status=exc.code,
            request_bytes=request_bytes,
            response_bytes=len(body_bytes),
            elapsed_seconds=elapsed_seconds,
        )
        body = body_bytes.decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except error.URLError as exc:
        elapsed_seconds = time.perf_counter() - started_at
        log_server_metric(
            enabled=log_metrics,
            step_name=step_name,
            method=method,
            url=url,
            status="url-error",
            request_bytes=request_bytes,
            response_bytes=0,
            elapsed_seconds=elapsed_seconds,
        )
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
    env_values = load_env(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Crop a masked region from an image, embed it, and match it against a Meilisearch texture index.")
    parser.add_argument("--image", type=Path, required=True, help="Input image file.")
    parser.add_argument("--mask", type=Path, required=True, help="Binary mask image, same size as the input image.")
    parser.add_argument("--margin", type=int, default=10, help="Pixel padding added around the mask bounding box.")
    parser.add_argument("--bbox-x1", type=int, help="Optional crop bbox x1 (inclusive) provided by caller.")
    parser.add_argument("--bbox-y1", type=int, help="Optional crop bbox y1 (inclusive) provided by caller.")
    parser.add_argument("--bbox-x2", type=int, help="Optional crop bbox x2 (exclusive) provided by caller.")
    parser.add_argument("--bbox-y2", type=int, help="Optional crop bbox y2 (exclusive) provided by caller.")
    parser.add_argument("--host", default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST"), help="Meilisearch host.")
    parser.add_argument("--key", default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"), help="Meilisearch API key.")
    parser.add_argument("--index", default=env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or DEFAULT_HSI_DINOV2_INDEX, help="Meilisearch candidate index to search.")
    parser.add_argument(
        "--method",
        default="hsi_dinov2",
        choices=["hsi_dinov2"],
        help="Embedding backend (only hsi_dinov2 is supported).",
    )
    parser.add_argument("--h-bins", type=int, default=36, help="Hue histogram bins for HSI prefilter.")
    parser.add_argument("--s-bins", type=int, default=16, help="Saturation histogram bins for HSI prefilter.")
    parser.add_argument("--i-bins", type=int, default=16, help="Intensity histogram bins for HSI prefilter.")
    parser.add_argument("--hsi-prefilter-limit", type=int, default=100, help="Number of HSI candidates kept for --method hsi_dinov2.")
    parser.add_argument("--hsi-dinov2-size", type=int, default=224, help="Output size for resized query crop in --method hsi_dinov2.")
    parser.add_argument("--limit", type=int, default=10, help="Number of matches to return.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for embedding inference.")
    parser.add_argument("--output", type=Path, help="Optional JSON file to save the result.")
    parser.add_argument("--filter-location", help="Optional selected location key/value filter from viewer.")
    parser.add_argument("--filter-application", help="Optional selected application key/value filter from viewer.")
    parser.add_argument("--filter-product-category", help="Optional selected product category key/value filter from viewer.")
    parser.add_argument("--filter-material-type", help="Optional selected material type key/value filter from viewer.")
    parser.add_argument("--filter-color", help="Optional selected color key/value filter from viewer.")
    parser.add_argument("--filter-brand-name", help="Optional selected brand name key/value filter from viewer.")
    parser.add_argument("--log-server-metrics", action="store_true", help="Log Meilisearch request/response sizes and latency for each HTTP call.")
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


def extract_max_fitting_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    binary = normalize_mask(mask)
    if not np.any(binary):
        raise ValueError("Mask is empty; no non-zero pixels were found.")

    height, width = binary.shape
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best_bbox = (0, 0, 1, 1)

    for row_idx in range(height):
        row = binary[row_idx]
        heights = np.where(row, heights + 1, 0)

        stack: list[tuple[int, int]] = []
        for col_idx in range(width + 1):
            current_height = int(heights[col_idx]) if col_idx < width else 0
            start = col_idx
            while stack and stack[-1][1] > current_height:
                left_col, popped_height = stack.pop()
                area = popped_height * (col_idx - left_col)
                if area > best_area:
                    x1 = left_col
                    x2 = col_idx
                    y2 = row_idx + 1
                    y1 = y2 - popped_height
                    best_bbox = (x1, y1, x2, y2)
                    best_area = area
                start = left_col
            if not stack or stack[-1][1] < current_height:
                stack.append((start, current_height))

    return best_bbox


def crop_image_from_mask(image: np.ndarray, mask: np.ndarray, margin: int = 10) -> tuple[np.ndarray, dict[str, int]]:
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image shape {image.shape[:2]} and mask shape {mask.shape[:2]} do not match.")
    if margin < 0:
        raise ValueError(f"Margin must be >= 0, got {margin}.")
    x1, y1, x2, y2 = extract_max_fitting_bbox(mask)
    if margin > 0:
        shrunk_x1 = x1 + margin
        shrunk_y1 = y1 + margin
        shrunk_x2 = x2 - margin
        shrunk_y2 = y2 - margin
        if shrunk_x1 < shrunk_x2 and shrunk_y1 < shrunk_y2:
            x1, y1, x2, y2 = shrunk_x1, shrunk_y1, shrunk_x2, shrunk_y2
    cropped = image[y1:y2, x1:x2]
    bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2 - x1, "height": y2 - y1}
    return cropped, bbox


def crop_image_from_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, dict[str, int]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"Provided bbox {bbox} is out of image bounds {width}x{height}.")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Provided bbox {bbox} is invalid.")
    cropped = image[y1:y2, x1:x2]
    bbox_dict = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2 - x1, "height": y2 - y1}
    return cropped, bbox_dict


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vector, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vector / norms


def compute_dino_embedding(image: Image.Image, model: AutoModel, processor: AutoImageProcessor, device: str) -> list[float]:
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


def compute_hsi_embedding(image: Image.Image, h_bins: int, s_bins: int, i_bins: int) -> list[float]:
    if h_bins <= 0 or s_bins <= 0 or i_bins <= 0:
        raise ValueError("HSI bin counts must be > 0.")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
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


def compute_resized_dino_query(
    image: Image.Image,
    output_size: int,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
) -> dict[str, Any]:
    if output_size <= 0:
        raise ValueError(f"output_size must be > 0, got {output_size}.")

    resized = image.convert("RGB").resize((output_size, output_size), resample=Image.Resampling.BICUBIC)
    try:
        emb_list = compute_dino_embedding(
            resized,
            model,
            processor,
            device,
        )
    finally:
        resized.close()
    query_vec = l2_normalize(np.asarray(emb_list, dtype=np.float32).reshape(1, -1))[0]
    return {"embedding": emb_list, "vector": query_vec}


def wait_for_task(
    host: str,
    api_key: str | None,
    task_uid: int,
    timeout_seconds: int = 600,
    poll_seconds: float = 0.5,
    *,
    log_metrics: bool = False,
) -> None:
    start = time.time()
    endpoint = f"{host.rstrip('/')}/tasks/{task_uid}"
    while True:
        task = http_json_with_metrics(
            "GET",
            endpoint,
            api_key=api_key,
            log_metrics=log_metrics,
            step_name=f"task {task_uid} poll",
        )
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


def configure_index_embedder(
    host: str,
    api_key: str | None,
    index_name: str,
    vector_dimensions: int,
    *,
    log_metrics: bool = False,
) -> None:
    settings = {
        "embedders": {
            "default": {
                "source": "userProvided",
                "dimensions": vector_dimensions,
            }
        }
    }
    response = http_json_with_metrics(
        "PATCH",
        f"{host.rstrip('/')}/indexes/{index_name}/settings",
        settings,
        api_key,
        log_metrics=log_metrics,
        step_name="update embedder settings",
    )
    wait_for_task(host, api_key, extract_task_uid(response, "update-settings"), log_metrics=log_metrics)


def search_meilisearch(
    host: str,
    api_key: str | None,
    index_name: str,
    vector: list[float],
    limit: int = 10,
    filter_expression: str | None = None,
    *,
    log_metrics: bool = False,
) -> dict[str, Any]:
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/search"
    payload = {
        "vector": vector,
        "limit": limit,
        "attributesToRetrieve": SEARCH_ATTRIBUTES_TO_RETRIEVE,
        "hybrid": {"semanticRatio": 1.0, "embedder": "default"},
        "showRankingScore": True,
    }
    if filter_expression:
        payload["filter"] = filter_expression
    try:
        return http_json_with_metrics(
            "POST",
            endpoint,
            payload,
            api_key,
            log_metrics=log_metrics,
            step_name="vector search",
        )
    except RuntimeError as exc:
        message = str(exc)
        if "Cannot find embedder with name `default`" not in message and "invalid_search_embedder" not in message:
            raise
        configure_index_embedder(host, api_key, index_name, len(vector), log_metrics=log_metrics)
        return http_json_with_metrics(
            "POST",
            endpoint,
            payload,
            api_key,
            log_metrics=log_metrics,
            step_name="vector search (retry)",
        )


def escape_filter_value(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def build_attribute_filter_expression_from_values(filter_values: dict[str, Any]) -> str | None:
    raw_filters: list[tuple[list[str], Any]] = [
        (["location"], filter_values.get("location")),
        (["application"], filter_values.get("application")),
        (["product_category", "productCategory"], filter_values.get("product_category")),
        (["material_type", "materialType"], filter_values.get("material_type")),
        (["color"], filter_values.get("color")),
        (["brand_name", "brandName"], filter_values.get("brand_name")),
    ]
    clauses: list[str] = []
    for fields, raw_value in raw_filters:
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if not value:
            continue
        escaped = escape_filter_value(value)
        per_field = [f'{field} = "{escaped}"' for field in fields]
        if len(per_field) == 1:
            clauses.append(per_field[0])
        else:
            clauses.append("(" + " OR ".join(per_field) + ")")
    if not clauses:
        return None
    return " AND ".join(clauses)


def build_attribute_filter_expression(args: argparse.Namespace) -> str | None:
    return build_attribute_filter_expression_from_values(
        {
            "location": args.filter_location,
            "application": args.filter_application,
            "product_category": args.filter_product_category,
            "material_type": args.filter_material_type,
            "color": args.filter_color,
            "brand_name": args.filter_brand_name,
        }
    )


def extract_numeric_score(hit: dict[str, Any]) -> float:
    raw_score = (
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
    if raw_score is None:
        return 0.0
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return 0.0


def build_best_hsi_score_map(hits: list[dict[str, Any]], limit: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for hit in hits[:limit]:
        raw = hit.get("product_id") or hit.get("id")
        if raw is None:
            continue
        product_id = str(raw)
        if not product_id:
            continue
        score = extract_numeric_score(hit)
        existing = scores.get(product_id)
        if existing is None or score > existing:
            scores[product_id] = score
    return scores


def format_hybrid_match(hit: dict[str, Any], rank: int) -> dict[str, Any]:
    formatted = format_match(hit, rank)
    formatted["hsi_score"] = hit.get("hsi_score")
    formatted["dinov2_score"] = hit.get("dinov2_score")
    formatted["combined_score"] = hit.get("combined_score")
    formatted["best_embedding_index"] = hit.get("best_embedding_index")
    formatted["best_embedding_image_name"] = hit.get("best_embedding_image_name")
    raw_embedding = hit.get("dinov2_embedding")
    if not isinstance(raw_embedding, list):
        raw_embedding = hit.get("embedding")
    if isinstance(raw_embedding, list):
        formatted["embedding"] = [float(value) for value in raw_embedding]
        formatted["embedding_dim"] = len(formatted["embedding"])
    return formatted


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
    image_url = hit.get("image_url") or hit.get("imageUrl") or build_texture_url(preview_path)
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
        "brandName": hit.get("brandName") or hit.get("brand_name"),
        "brand_name": hit.get("brand_name") or hit.get("brandName"),
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


class MaskedRegionTextureMatcher:
    def __init__(
        self,
        *,
        host: str | None = None,
        api_key: str | None = None,
        index_name: str = DEFAULT_HSI_DINOV2_INDEX,
        method: str = "hsi_dinov2",
        h_bins: int = 36,
        s_bins: int = 16,
        i_bins: int = 16,
        hsi_prefilter_limit: int = 100,
        hsi_dinov2_size: int = 224,
        limit: int = 10,
        device: str = "auto",
        log_server_metrics: bool = False,
    ):
        env_values = load_env(PROJECT_ROOT / ".env")
        self.host = (host or env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or DEFAULT_MEILI_HOST).strip()
        self.api_key = str(api_key).strip() if api_key else None
        self.index_name = index_name
        self.method = method
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.i_bins = i_bins
        self.hsi_prefilter_limit = hsi_prefilter_limit
        self.hsi_dinov2_size = hsi_dinov2_size
        self.limit = limit
        self.device = choose_device(device)
        self.log_server_metrics = bool(log_server_metrics)
        if self.hsi_prefilter_limit <= 0:
            raise ValueError(f"hsi_prefilter_limit must be > 0, got {self.hsi_prefilter_limit}.")
        if self.hsi_dinov2_size <= 0:
            raise ValueError(f"hsi_dinov2_size must be > 0, got {self.hsi_dinov2_size}.")
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME_DINOV2, token=hf_token)
        self.model = AutoModel.from_pretrained(MODEL_NAME_DINOV2, token=hf_token)
        self.model.to(self.device)
        self.model.eval()

    def _build_matches(
        self,
        *,
        hsi_hits: list[dict[str, Any]],
        embedding: list[float],
        query_vec: np.ndarray,
    ) -> list[dict[str, Any]]:
        hsi_scores = build_best_hsi_score_map(hsi_hits, self.hsi_prefilter_limit)
        best_by_product_id: dict[str, dict[str, Any]] = {}
        for doc in hsi_hits:
            raw_product_id = doc.get("product_id") or doc.get("id")
            if raw_product_id is None:
                continue
            product_id = str(raw_product_id)
            hsi_score = hsi_scores.get(product_id)
            if hsi_score is None:
                continue

            candidate_vectors: list[tuple[int | None, list[Any]]] = []
            raw_dinov2_vectors = doc.get("dinov2_embedding_vectors")
            if isinstance(raw_dinov2_vectors, list) and raw_dinov2_vectors:
                if all(isinstance(value, (int, float)) for value in raw_dinov2_vectors):
                    candidate_vectors.append((0, raw_dinov2_vectors))
                else:
                    for vector_index, candidate_vector in enumerate(raw_dinov2_vectors):
                        if isinstance(candidate_vector, list) and candidate_vector:
                            candidate_vectors.append((vector_index, candidate_vector))

            if not candidate_vectors:
                raw_doc_embedding = doc.get("embedding")
                if isinstance(raw_doc_embedding, list) and raw_doc_embedding:
                    candidate_vectors.append((None, raw_doc_embedding))
            if not candidate_vectors:
                continue

            best_dino_score_for_doc: float | None = None
            best_vector_index_for_doc: int | None = None
            for vector_index, candidate_vector in candidate_vectors:
                doc_vec = np.asarray(candidate_vector, dtype=np.float32).reshape(-1)
                if doc_vec.size != len(embedding):
                    continue
                doc_vec = l2_normalize(doc_vec.reshape(1, -1))[0]
                score = float(np.dot(query_vec, doc_vec))
                if best_dino_score_for_doc is None or score > best_dino_score_for_doc:
                    best_dino_score_for_doc = score
                    best_vector_index_for_doc = vector_index

            if best_dino_score_for_doc is None:
                continue
            dino_score = max(0.0, float(best_dino_score_for_doc))
            hsi_score_value = float(hsi_score)
            combined_score = dino_score * hsi_score_value

            reranked = dict(doc)
            reranked["hsi_score"] = hsi_score_value
            reranked["dinov2_score"] = dino_score
            reranked["combined_score"] = combined_score
            reranked["score"] = combined_score
            reranked["rankingScore"] = combined_score
            reranked["best_embedding_index"] = best_vector_index_for_doc
            image_names = doc.get("image_names")
            if (
                isinstance(image_names, list)
                and best_vector_index_for_doc is not None
                and 0 <= best_vector_index_for_doc < len(image_names)
            ):
                reranked["best_embedding_image_name"] = image_names[best_vector_index_for_doc]

            existing = best_by_product_id.get(product_id)
            if existing is None or float(existing["combined_score"]) < combined_score:
                best_by_product_id[product_id] = reranked

        if hsi_hits and not best_by_product_id:
            raise RuntimeError(
                "HSI prefilter returned candidates, but none could be reranked with a compatible DINOv2 embedding. "
                f"Checked {len(hsi_hits)} candidate documents in index '{self.index_name}'. "
                "Verify product_id alignment and embedding vector dimensions."
            )

        hits = sorted(
            best_by_product_id.values(),
            key=lambda item: float(item["combined_score"]),
            reverse=True,
        )[: self.limit]
        return [format_hybrid_match(hit, index) for index, hit in enumerate(hits, start=1)]

    def _run_match(
        self,
        *,
        image_array: np.ndarray,
        mask_array: np.ndarray,
        image_label: str,
        mask_label: str,
        margin: int,
        provided_bbox: tuple[int, int, int, int] | None,
        filter_values: dict[str, Any] | None,
    ) -> dict[str, Any]:
        run_started_at = time.perf_counter()
        if provided_bbox is not None:
            cropped, bbox = crop_image_from_bbox(image_array, provided_bbox)
        else:
            cropped, bbox = crop_image_from_mask(image_array, mask_array, margin=margin)

        cropped_image = Image.fromarray(cropped.astype(np.uint8))
        try:
            hsi_embedding_started_at = time.perf_counter()
            hsi_embedding = compute_hsi_embedding(
                image=cropped_image,
                h_bins=self.h_bins,
                s_bins=self.s_bins,
                i_bins=self.i_bins,
            )
            hsi_embedding_seconds = time.perf_counter() - hsi_embedding_started_at

            attribute_filter_expression = build_attribute_filter_expression_from_values(filter_values or {})
            hsi_search_started_at = time.perf_counter()
            hsi_result = search_meilisearch(
                self.host,
                self.api_key,
                self.index_name,
                hsi_embedding,
                limit=self.hsi_prefilter_limit,
                filter_expression=attribute_filter_expression,
                log_metrics=self.log_server_metrics,
            )
            hsi_search_seconds = time.perf_counter() - hsi_search_started_at
            hsi_hits = hsi_result.get("hits", []) if isinstance(hsi_result, dict) else []

            dino_query_seconds = 0.0
            rerank_seconds = 0.0
            if not hsi_hits:
                embedding = hsi_embedding
                matches: list[dict[str, Any]] = []
            else:
                dino_query_started_at = time.perf_counter()
                query_data = compute_resized_dino_query(
                    image=cropped_image,
                    output_size=self.hsi_dinov2_size,
                    model=self.model,
                    processor=self.image_processor,
                    device=self.device,
                )
                dino_query_seconds = time.perf_counter() - dino_query_started_at
                embedding = query_data["embedding"]
                rerank_started_at = time.perf_counter()
                matches = self._build_matches(
                    hsi_hits=hsi_hits,
                    embedding=embedding,
                    query_vec=query_data["vector"],
                )
                rerank_seconds = time.perf_counter() - rerank_started_at

            return {
                "image": image_label,
                "mask": mask_label,
                "bbox": bbox,
                "method": self.method,
                "embedding_dim": len(embedding),
                "timing": {
                    "hsi_embedding_seconds": hsi_embedding_seconds,
                    "hsi_search_seconds": hsi_search_seconds,
                    "dino_query_seconds": dino_query_seconds,
                    "rerank_seconds": rerank_seconds,
                    "total_seconds": time.perf_counter() - run_started_at,
                },
                "matches": matches,
            }
        finally:
            cropped_image.close()

    def match_from_paths(
        self,
        *,
        image_path: Path,
        mask_path: Path,
        margin: int = 10,
        provided_bbox: tuple[int, int, int, int] | None = None,
        filter_values: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")
        try:
            return self._run_match(
                image_array=np.asarray(image),
                mask_array=np.asarray(mask_image),
                image_label=str(image_path),
                mask_label=str(mask_path),
                margin=margin,
                provided_bbox=provided_bbox,
                filter_values=filter_values,
            )
        finally:
            image.close()
            mask_image.close()

    def match_from_arrays(
        self,
        *,
        image_array: np.ndarray,
        mask_array: np.ndarray,
        image_label: str = "<in-memory-image>",
        mask_label: str = "<in-memory-mask>",
        margin: int = 10,
        provided_bbox: tuple[int, int, int, int] | None = None,
        filter_values: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._run_match(
            image_array=image_array,
            mask_array=mask_array,
            image_label=image_label,
            mask_label=mask_label,
            margin=margin,
            provided_bbox=provided_bbox,
            filter_values=filter_values,
        )


def main() -> int:
    args = parse_args()
    bbox_args = [args.bbox_x1, args.bbox_y1, args.bbox_x2, args.bbox_y2]
    provided_bbox: tuple[int, int, int, int] | None = None
    if any(value is not None for value in bbox_args):
        if not all(value is not None for value in bbox_args):
            raise ValueError("When providing bbox coordinates, all of --bbox-x1/--bbox-y1/--bbox-x2/--bbox-y2 are required.")
        provided_bbox = (int(args.bbox_x1), int(args.bbox_y1), int(args.bbox_x2), int(args.bbox_y2))

    matcher = MaskedRegionTextureMatcher(
        host=args.host,
        api_key=args.key,
        index_name=args.index,
        method=args.method,
        h_bins=args.h_bins,
        s_bins=args.s_bins,
        i_bins=args.i_bins,
        hsi_prefilter_limit=args.hsi_prefilter_limit,
        hsi_dinov2_size=args.hsi_dinov2_size,
        limit=args.limit,
        device=args.device,
        log_server_metrics=args.log_server_metrics,
    )
    summary = matcher.match_from_paths(
        image_path=args.image,
        mask_path=args.mask,
        margin=args.margin,
        provided_bbox=provided_bbox,
        filter_values={
            "location": args.filter_location,
            "application": args.filter_application,
            "product_category": args.filter_product_category,
            "material_type": args.filter_material_type,
            "color": args.filter_color,
            "brand_name": args.filter_brand_name,
        },
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_json = json.dumps(summary, ensure_ascii=True, indent=2) + "\n"
    sys.stdout.write(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
