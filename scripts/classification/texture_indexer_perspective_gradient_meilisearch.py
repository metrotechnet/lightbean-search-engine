#!/usr/bin/env python3
"""Index perspective-augmented gradient texture embeddings into Meilisearch.

Pipeline:
1) Convert texture image to grayscale gradient magnitude (Sobel).
2) Generate 3D perspective projections for yaw/pitch combinations.
3) Save projections.
4) Embed projections and index in Meilisearch, linked to original product_id.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import ssl
import time
from pathlib import Path
from typing import Any
from urllib import error, request

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-small"
DEFAULT_PRODUCT_EXPORT = "texture/lightbeans_main_page_textures.json"
DEFAULT_TEXTURE_DIR = "texture/main_page"
DEFAULT_PROJECTION_DIR = "texture/perspective_gradients"
MEILI_INDEX = "lightbeans-texture2-dinov2-gradient-perspective"


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
    payload = {"uid": index_name, "primaryKey": "id"}
    try:
        response = http_json("POST", endpoint, payload, api_key)
        wait_for_task(host, api_key, extract_task_uid(response, "create-index"))
    except RuntimeError as exc:
        message = str(exc).lower()
        if "already exists" not in message and "index_already_exists" not in message:
            raise

    settings: dict[str, Any] = {
        "searchableAttributes": ["name", "link", "image_url", "image_name", "product_id"],
        "filterableAttributes": ["id", "product_id", "yaw", "pitch", "roll"],
        "sortableAttributes": ["product_id", "yaw", "pitch"],
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


def angle_tag(angle: float) -> str:
    if float(angle).is_integer():
        return str(int(angle))
    return str(angle).replace(".", "p")


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    ry = np.array([[math.cos(yaw), 0.0, math.sin(yaw)], [0.0, 1.0, 0.0], [-math.sin(yaw), 0.0, math.cos(yaw)]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(pitch), -math.sin(pitch)], [0.0, math.sin(pitch), math.cos(pitch)]], dtype=np.float32)
    rz = np.array([[math.cos(roll), -math.sin(roll), 0.0], [math.sin(roll), math.cos(roll), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ ry @ rx


def extract_max_fitting_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    binary = mask.astype(bool)
    if not np.any(binary):
        raise ValueError("Mask is empty; no valid projection region.")

    height, width = binary.shape
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best_bbox = (0, 0, width, height)

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


def compute_gradient_magnitude(gray_uint8: np.ndarray, sobel_ksize: int, grad_norm_max: float) -> np.ndarray:
    gray = gray_uint8.astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    magnitude = cv2.magnitude(grad_x, grad_y)
    normalized = np.clip(magnitude / float(grad_norm_max), 0.0, 1.0)
    return normalized


def project_gradient(
    gradient: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    output_width: int,
    output_height: int,
    focal_px: float,
    distance_px: float,
    crop_valid_region: bool,
) -> tuple[np.ndarray, np.ndarray]:
    in_h, in_w = gradient.shape
    src = np.array([[0.0, 0.0], [in_w - 1.0, 0.0], [in_w - 1.0, in_h - 1.0], [0.0, in_h - 1.0]], dtype=np.float32)

    max_dim = max(in_w, in_h)
    if max_dim <= 0:
        raise ValueError("Invalid input image size for projection.")
    half_w = (in_w / max_dim) * 0.5
    half_h = (in_h / max_dim) * 0.5
    plane_pts = np.array(
        [[-half_w, -half_h, 0.0], [half_w, -half_h, 0.0], [half_w, half_h, 0.0], [-half_w, half_h, 0.0]],
        dtype=np.float32,
    )
    rot = rotation_matrix(yaw, pitch, roll)
    rotated = (rot @ plane_pts.T).T
    rotated[:, 2] += distance_px

    cx = output_width / 2.0
    cy = output_height 
    projected = np.zeros((4, 2), dtype=np.float32)
    for idx, pt in enumerate(rotated):
        projected[idx, 0] = focal_px * (pt[0] / pt[2]) + cx
        projected[idx, 1] = focal_px * (pt[1] / pt[2]) + cy

    h_matrix = cv2.getPerspectiveTransform(src, projected)
    warped = cv2.warpPerspective(
        gradient,
        h_matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    valid_src = np.ones((in_h, in_w), dtype=np.uint8)
    valid_mask = cv2.warpPerspective(
        valid_src,
        h_matrix,
        (output_width, output_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_binary = valid_mask > 0

    if not np.any(valid_binary):
        fallback = cv2.resize(gradient, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
        return fallback, np.ones((output_height, output_width), dtype=bool)

    if crop_valid_region:
        try:
            x1, y1, x2, y2 = extract_max_fitting_bbox(valid_binary)
        except ValueError:
            ys, xs = np.nonzero(valid_binary)
            if xs.size == 0 or ys.size == 0:
                fallback = cv2.resize(gradient, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
                return fallback, np.ones((output_height, output_width), dtype=bool)
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
        cropped = warped[y1:y2, x1:x2]
        if cropped.size > 0:
            warped = cv2.resize(cropped, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
            valid_binary = np.ones((output_height, output_width), dtype=bool)

    return warped, valid_binary


def to_uint8_image(normalized_float: np.ndarray) -> np.ndarray:
    return np.clip(np.round(normalized_float * 255.0), 0, 255).astype(np.uint8)


def compute_embedding_from_projection(
    projection_uint8: np.ndarray,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
) -> list[float]:
    image = Image.fromarray(projection_uint8, mode="L").convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embedding = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            embedding = outputs.last_hidden_state[:, 0, :]
        else:
            raise RuntimeError("Model output does not expose a usable embedding.")
    vector = l2_normalize(embedding.detach().cpu().numpy().astype(np.float32))[0]
    return [float(v) for v in vector.tolist()]


def build_document(
    product: dict[str, Any],
    projection_path: Path,
    embedding: list[float],
    yaw: float,
    pitch: float,
    roll: float,
) -> dict[str, Any]:
    base_product_id = product.get("id") or product.get("image_name") or product.get("link") or projection_path.stem
    product_id = sanitize_meilisearch_id(base_product_id)
    projection_id = sanitize_meilisearch_id(f"{product_id}_yaw{angle_tag(yaw)}_pitch{angle_tag(pitch)}_roll{angle_tag(roll)}")
    return {
        "id": projection_id,
        "product_id": product_id,
        "name": product.get("name"),
        "link": product.get("link"),
        "image_url": product.get("image_url") or "",
        "image_name": product.get("image_name") or projection_path.name,
        "projection_path": projection_path.as_posix(),
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "embedding": embedding,
        "_vectors": {"default": embedding},
    }


def upload_documents(host: str, api_key: str | None, index_name: str, documents: list[dict[str, Any]], batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/documents"
    if not documents:
        return
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        response = http_json("POST", endpoint, batch, api_key, retries=5, retry_delay_seconds=2.0)
        wait_for_task(host, api_key, extract_task_uid(response, "upload-documents"))


def search_meilisearch(host: str, api_key: str | None, index_name: str, vector: list[float], limit: int = 10) -> dict[str, Any]:
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/search"
    payload = {
        "vector": vector,
        "limit": limit,
        "attributesToRetrieve": ["id", "product_id", "name", "link", "image_url", "image_name", "yaw", "pitch", "roll"],
        "hybrid": {"semanticRatio": 1.0, "embedder": "default"},
        "showRankingScore": True,
    }
    return http_json("POST", endpoint, payload, api_key)


def search_query_image(
    query_image: Path,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
    sobel_ksize: int,
    grad_norm_max: float,
    host: str,
    key: str | None,
    index: str,
    limit: int,
) -> dict[str, Any]:
    pil = Image.open(query_image).convert("L")
    gray = np.asarray(pil, dtype=np.uint8)
    pil.close()
    gradient = compute_gradient_magnitude(gray, sobel_ksize=sobel_ksize, grad_norm_max=grad_norm_max)
    vector = compute_embedding_from_projection(to_uint8_image(gradient), model, processor, device)
    result = search_meilisearch(host, key, index, vector, limit=limit)
    hits = result.get("hits", []) if isinstance(result, dict) else []
    return {
        "image": str(query_image),
        "embedding_dim": len(vector),
        "matches": hits[:limit],
    }


def parse_angle_list(value: str) -> list[float]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise ValueError("Angle list cannot be empty.")
    return [float(item) for item in parts]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    env_values = load_env(project_root / ".env")
    parser = argparse.ArgumentParser(description="Index perspective gradient texture embeddings into Meilisearch.")
    parser.add_argument("--product-export", type=Path, default=Path(DEFAULT_PRODUCT_EXPORT), help="Product export JSON file.")
    parser.add_argument("--textures-dir", type=Path, default=Path(DEFAULT_TEXTURE_DIR), help="Directory containing downloaded texture files.")
    parser.add_argument("--mapping-json", type=Path, default=None, help="Optional file-to-product mapping JSON.")
    parser.add_argument("--projection-dir", type=Path, default=Path(DEFAULT_PROJECTION_DIR), help="Output directory for projection images.")
    parser.add_argument("--host", default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700", help="Meilisearch host.")
    parser.add_argument("--key", default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"), help="Meilisearch API key.")
    parser.add_argument("--index", default=env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or MEILI_INDEX, help="Target Meilisearch index name.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Embedding model name.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for embedding.")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N products (0=all).")
    parser.add_argument("--batch-size", type=int, default=100, help="Upload batch size.")
    parser.add_argument("--yaw-angles", default="-30,0,30", help="Comma-separated yaw angles in degrees.")
    parser.add_argument("--pitch-angles", default="-30,0,30", help="Comma-separated pitch angles in degrees.")
    parser.add_argument("--roll", type=float, default=0.0, help="Roll angle in degrees.")
    parser.add_argument("--output-width", type=int, default=448, help="Projection output width.")
    parser.add_argument("--output-height", type=int, default=448, help="Projection output height.")
    parser.add_argument("--focal-length", type=float, default=650.0, help="Virtual camera focal length in pixels.")
    parser.add_argument("--camera-distance", type=float, default=2.5, help="Plane distance from camera in normalized world units.")
    parser.add_argument("--sobel-ksize", type=int, default=3, choices=[1, 3, 5, 7], help="Sobel kernel size.")
    parser.add_argument("--grad-norm-max", type=float, default=11.313708, help="Fixed normalization maximum for gradient magnitude.")
    parser.add_argument("--perspective-augmentation", action=argparse.BooleanOptionalAction, default=True, help="Enable 3x3 yaw/pitch projections.")
    parser.add_argument("--crop-valid-region", action=argparse.BooleanOptionalAction, default=True, help="Crop valid projected region before resize.")
    parser.add_argument("--query-image", type=Path, help="Optional query image path to run a recognition search.")
    parser.add_argument("--search-limit", type=int, default=10, help="Search hits for --query-image.")
    parser.add_argument("--search-output", type=Path, help="Optional output JSON path for search payload.")
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
    if not args.product_export.exists():
        raise FileNotFoundError(f"Product export not found: {args.product_export}")
    if not args.textures_dir.exists():
        raise FileNotFoundError(f"Textures directory not found: {args.textures_dir}")

    args.projection_dir.mkdir(parents=True, exist_ok=True)
    yaw_angles = parse_angle_list(args.yaw_angles)
    pitch_angles = parse_angle_list(args.pitch_angles)
    if not args.perspective_augmentation:
        yaw_angles = [0.0]
        pitch_angles = [0.0]

    device = choose_device(args.device)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    processor = AutoImageProcessor.from_pretrained(args.model_name, token=hf_token)
    model = AutoModel.from_pretrained(args.model_name, token=hf_token)
    model.to(device)
    model.eval()

    products = load_product_export(args.product_export)
    if args.limit > 0:
        products = products[: args.limit]

    documents: list[dict[str, Any]] = []
    skipped = 0
    for record in products:
        product_id_raw = record.get("id") or record.get("image_name") or record.get("link") or record.get("image_url")
        if product_id_raw is None:
            continue
        if "id" not in record:
            record["id"] = sanitize_meilisearch_id(product_id_raw)

        texture_path = resolve_texture_path(record, args.textures_dir, args.mapping_json)
        if texture_path is None or not texture_path.exists():
            skipped += 1
            continue

        source_image = Image.open(texture_path).convert("L")
        gray = np.asarray(source_image, dtype=np.uint8)
        source_image.close()
        gradient = compute_gradient_magnitude(gray, sobel_ksize=args.sobel_ksize, grad_norm_max=args.grad_norm_max)
        texture_stem = texture_path.stem
        product_projection_dir = args.projection_dir / sanitize_meilisearch_id(record.get("id"))
        product_projection_dir.mkdir(parents=True, exist_ok=True)

        for yaw in yaw_angles:
            for pitch in pitch_angles:
                projected, _valid_mask = project_gradient(
                    gradient=gradient,
                    yaw=yaw,
                    pitch=pitch,
                    roll=args.roll,
                    output_width=args.output_width,
                    output_height=args.output_height,
                    focal_px=args.focal_length,
                    distance_px=args.camera_distance,
                    crop_valid_region=args.crop_valid_region,
                )
                projection_name = f"{texture_stem}_yaw{angle_tag(yaw)}_pitch{angle_tag(pitch)}.png"
                projection_path = product_projection_dir / projection_name
                projection_uint8 = to_uint8_image(projected)
                cv2.imwrite(str(projection_path), projection_uint8)

                embedding = compute_embedding_from_projection(projection_uint8, model, processor, device)
                documents.append(build_document(record, projection_path, embedding, yaw=yaw, pitch=pitch, roll=args.roll))

    vector_dimensions = len(documents[0]["embedding"]) if documents else None
    ensure_index(args.host, args.key, args.index, vector_dimensions=vector_dimensions)
    upload_documents(args.host, args.key, args.index, documents, batch_size=args.batch_size)
    print(f"Indexed {len(documents)} perspective-gradient embeddings in Meilisearch index '{args.index}'.")
    print(f"Skipped {skipped} products without a usable texture file.")

    if args.query_image:
        if not args.query_image.exists():
            raise FileNotFoundError(f"Query image not found: {args.query_image}")
        search_payload = search_query_image(
            query_image=args.query_image,
            model=model,
            processor=processor,
            device=device,
            sobel_ksize=args.sobel_ksize,
            grad_norm_max=args.grad_norm_max,
            host=args.host,
            key=args.key,
            index=args.index,
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
