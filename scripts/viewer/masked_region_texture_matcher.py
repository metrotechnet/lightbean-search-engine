#!/usr/bin/env python3
"""Match a masked image region against texture embedding indexes in Meilisearch.

Workflow:
1. Load an input image and a binary mask.
2. Find the largest axis-aligned rectangle fully contained in the mask.
3. Crop the image to that box and compute an embedding (DINOv2, DINOv3, Gram, HSI, perspective-gradient, DINOv2 texture-stats).
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
from urllib.parse import urljoin

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME_DINOV2 = "facebook/dinov2-small"
MODEL_NAME_DINOV3 = "facebook/dinov3-vitb16-pretrain-lvd1689m"
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
    project_root = Path(__file__).resolve().parents[2]
    env_values = load_env(project_root / ".env")
    parser = argparse.ArgumentParser(description="Crop a masked region from an image, embed it, and match it against a Meilisearch texture index.")
    parser.add_argument("--image", type=Path, required=True, help="Input image file.")
    parser.add_argument("--mask", type=Path, required=True, help="Binary mask image, same size as the input image.")
    parser.add_argument("--margin", type=int, default=10, help="Pixel padding added around the mask bounding box.")
    parser.add_argument("--bbox-x1", type=int, help="Optional crop bbox x1 (inclusive) provided by caller.")
    parser.add_argument("--bbox-y1", type=int, help="Optional crop bbox y1 (inclusive) provided by caller.")
    parser.add_argument("--bbox-x2", type=int, help="Optional crop bbox x2 (exclusive) provided by caller.")
    parser.add_argument("--bbox-y2", type=int, help="Optional crop bbox y2 (exclusive) provided by caller.")
    parser.add_argument("--host", default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700", help="Meilisearch host.")
    parser.add_argument("--key", default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"), help="Meilisearch API key.")
    parser.add_argument("--index", default=env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or DEFAULT_INDEX, help="Meilisearch index to search.")
    parser.add_argument(
        "--method",
        default="dinov2",
        choices=["dinov2", "dinov3", "gram", "hsi", "perspective", "hsi_dinov2", "dinov2_texture_faiss"],
        help="Embedding backend.",
    )
    parser.add_argument("--dinov3-model-name", default=MODEL_NAME_DINOV3, help="DINOv3 model name for --method dinov3.")
    parser.add_argument("--patch-pool", default="mean", choices=["mean", "max", "cls"], help="DINOv3 patch pooling mode.")
    parser.add_argument("--lbp-points", type=int, default=8, help="LBP neighborhood points for --method gram.")
    parser.add_argument("--lbp-radius", type=int, default=1, help="LBP radius for --method gram.")
    parser.add_argument("--gram-layer-index", type=int, default=8, help="VGG19 feature layer index for --method gram.")
    parser.add_argument("--h-bins", type=int, default=36, help="Hue histogram bins for --method hsi.")
    parser.add_argument("--s-bins", type=int, default=16, help="Saturation histogram bins for --method hsi.")
    parser.add_argument("--i-bins", type=int, default=16, help="Intensity histogram bins for --method hsi.")
    parser.add_argument("--hsi-prefilter-index", default="lightbeans-texture2-hsi", help="HSI prefilter index for --method hsi_dinov2.")
    parser.add_argument("--hsi-prefilter-limit", type=int, default=100, help="Number of HSI candidates kept for --method hsi_dinov2.")
    parser.add_argument("--hsi-dinov2-scales", default="0.5,1.0,1.5,2.0", help="Comma-separated query scales for --method hsi_dinov2.")
    parser.add_argument("--hsi-dinov2-rotations", default="0,45,90,135,180", help="Comma-separated query rotations for --method hsi_dinov2.")
    parser.add_argument("--hsi-dinov2-size", type=int, default=224, help="Output size for scaled/rotated queries in --method hsi_dinov2.")
    parser.add_argument("--perspective-model-name", default=MODEL_NAME_DINOV2, help="Embedding model name for --method perspective.")
    parser.add_argument("--yaw-angles", default="-30,0,30", help="Comma-separated yaw angles for --method perspective.")
    parser.add_argument("--pitch-angles", default="-30,0,30", help="Comma-separated pitch angles for --method perspective.")
    parser.add_argument("--perspective-roll", type=float, default=0.0, help="Roll angle for --method perspective.")
    parser.add_argument("--perspective-output-width", type=int, default=448, help="Projection output width for --method perspective.")
    parser.add_argument("--perspective-output-height", type=int, default=448, help="Projection output height for --method perspective.")
    parser.add_argument("--perspective-focal-length", type=float, default=650.0, help="Virtual camera focal length for --method perspective.")
    parser.add_argument("--perspective-camera-distance", type=float, default=2.5, help="Plane distance for --method perspective.")
    parser.add_argument("--sobel-ksize", type=int, default=3, choices=[1, 3, 5, 7], help="Sobel kernel size for --method perspective.")
    parser.add_argument("--grad-norm-max", type=float, default=11.313708, help="Fixed gradient normalization max for --method perspective.")
    parser.add_argument("--perspective-crop-valid-region", action=argparse.BooleanOptionalAction, default=True, help="Crop valid projected area for --method perspective.")
    parser.add_argument("--faiss-index-root", type=Path, default=Path("texture/indexes/dinov2_faiss"), help="Local FAISS index directory for --method dinov2_texture_faiss.")
    parser.add_argument("--faiss-embedding-mode", default="texture_stats", choices=["cls", "mean", "texture_stats"], help="Embedding mode for --method dinov2_texture_faiss.")
    parser.add_argument("--faiss-input-size", type=int, default=224, help="Input size used by --method dinov2_texture_faiss (must match index build config).")
    parser.add_argument("--faiss-batch-size", type=int, default=16, help="Batch size for --method dinov2_texture_faiss.")
    parser.add_argument("--limit", type=int, default=10, help="Number of matches to return.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for embedding inference.")
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


def compute_lbp_histogram(gray: np.ndarray, points: int = 8, radius: int = 1) -> np.ndarray:
    if points <= 0:
        raise ValueError("LBP points must be > 0.")
    if points > 16:
        raise ValueError("LBP points > 16 creates very large histograms; choose <= 16.")
    if radius <= 0:
        raise ValueError("LBP radius must be > 0.")

    h, w = gray.shape
    padded = np.pad(gray, radius, mode="edge")
    center = padded[radius : radius + h, radius : radius + w]
    lbp = np.zeros((h, w), dtype=np.uint32)

    for bit in range(points):
        theta = 2.0 * np.pi * bit / points
        dy = int(round(-radius * np.sin(theta)))
        dx = int(round(radius * np.cos(theta)))
        neighbor = padded[radius + dy : radius + dy + h, radius + dx : radius + dx + w]
        lbp |= ((neighbor >= center).astype(np.uint32) << bit)

    bins = 1 << points
    hist = np.bincount(lbp.ravel(), minlength=bins).astype(np.float32)
    hist_sum = float(hist.sum())
    if hist_sum > 0:
        hist /= hist_sum
    return hist


class VGGGramExtractor(torch.nn.Module):
    def __init__(self, layer_index: int = 8):
        super().__init__()
        weights = VGG19_Weights.IMAGENET1K_V1
        model = vgg19(weights=weights)
        features = list(model.features.children())
        if layer_index < 0 or layer_index >= len(features):
            raise ValueError(f"layer_index must be in [0, {len(features) - 1}], got {layer_index}.")
        self.features = torch.nn.Sequential(*features[: layer_index + 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.features(x)
        b, c, h, w = fmap.shape
        fm = fmap.view(b, c, h * w)
        gram = torch.bmm(fm, fm.transpose(1, 2)) / float(c * h * w)
        return gram


def gram_upper_triangle(gram: np.ndarray) -> np.ndarray:
    c, _ = gram.shape
    tri = np.triu_indices(c)
    return gram[tri].astype(np.float32)


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


def compute_dino_embedding(image: Image.Image, model: AutoModel, processor: AutoImageProcessor, device: str, method: str, patch_pooling: str) -> list[float]:
    pixel_values = processor(images=image, return_tensors="pt")
    pixel_values = {key: value.to(device) for key, value in pixel_values.items()}
    with torch.no_grad():
        outputs = model(**pixel_values)

        if hasattr(outputs, "last_hidden_state"):
            if method == "dinov3":
                embedding = patch_pool(outputs.last_hidden_state, patch_pooling)
            else:
                embedding = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embedding = outputs.pooler_output
        else:
            raise RuntimeError(f"{method} output does not expose a usable embedding.")

    embedding = l2_normalize(embedding.detach().cpu().numpy().astype(np.float32))[0]
    return [float(value) for value in embedding.tolist()]


def compute_gram_embedding(
    image: Image.Image,
    gram_extractor: VGGGramExtractor,
    image_transform: transforms.Compose,
    device: str,
    lbp_points: int,
    lbp_radius: int,
) -> list[float]:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    lbp_hist = compute_lbp_histogram(gray, points=lbp_points, radius=lbp_radius)

    tensor = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        gram = gram_extractor(tensor)[0].detach().cpu().numpy()
    gram_vec = gram_upper_triangle(gram)

    feature = np.concatenate([lbp_hist, gram_vec], axis=0).astype(np.float32)
    feature = l2_normalize(feature.reshape(1, -1))[0]
    return [float(v) for v in feature.tolist()]


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


def compute_scaled_rotated_dino_queries(
    image: Image.Image,
    scales: list[float],
    rotations: list[float],
    output_size: int,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
) -> list[dict[str, Any]]:
    if output_size <= 0:
        raise ValueError(f"output_size must be > 0, got {output_size}.")
    if not scales:
        raise ValueError("At least one scale is required for hsi_dinov2 query transforms.")
    if not rotations:
        raise ValueError("At least one rotation is required for hsi_dinov2 query transforms.")
    if any(scale <= 0 for scale in scales):
        raise ValueError(f"All scales must be > 0, got {scales}.")

    resample = Image.Resampling.BICUBIC
    base = image.convert("RGB")
    results: list[dict[str, Any]] = []
    try:
        base_width, base_height = base.size
        for scale in scales:
            scaled_width = max(1, int(round(base_width * float(scale))))
            scaled_height = max(1, int(round(base_height * float(scale))))
            scaled = base.resize((scaled_width, scaled_height), resample=resample)
            try:
                mask = Image.new("L", scaled.size, 255)
                try:
                    for rotation in rotations:
                        rotated = scaled.rotate(float(rotation), resample=resample, expand=True, fillcolor=(0, 0, 0))
                        rotated_mask = mask.rotate(float(rotation), resample=Image.Resampling.NEAREST, expand=True, fillcolor=0)
                        try:
                            valid = np.asarray(rotated_mask, dtype=np.uint8) > 0
                            try:
                                x1, y1, x2, y2 = extract_max_fitting_bbox(valid)
                            except ValueError:
                                x1, y1, x2, y2 = 0, 0, rotated.width, rotated.height

                            side = min(x2 - x1, y2 - y1)
                            if side <= 0:
                                continue
                            cx = x1 + (x2 - x1) // 2
                            cy = y1 + (y2 - y1) // 2
                            half = side // 2
                            crop_x1 = max(0, cx - half)
                            crop_y1 = max(0, cy - half)
                            crop_x2 = crop_x1 + side
                            crop_y2 = crop_y1 + side
                            if crop_x2 > rotated.width:
                                crop_x2 = rotated.width
                                crop_x1 = max(0, crop_x2 - side)
                            if crop_y2 > rotated.height:
                                crop_y2 = rotated.height
                                crop_y1 = max(0, crop_y2 - side)

                            transformed = rotated.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                            try:
                                transformed_resized = transformed.resize((output_size, output_size), resample=resample)
                                try:
                                    emb_list = compute_dino_embedding(
                                        transformed_resized,
                                        model,
                                        processor,
                                        device,
                                        method="dinov2",
                                        patch_pooling="mean",
                                    )
                                finally:
                                    transformed_resized.close()
                            finally:
                                transformed.close()
                            query_vec = l2_normalize(np.asarray(emb_list, dtype=np.float32).reshape(1, -1))[0]
                            results.append(
                                {
                                    "scale": float(scale),
                                    "rotation": float(rotation),
                                    "embedding": emb_list,
                                    "vector": query_vec,
                                }
                            )
                        finally:
                            rotated.close()
                            rotated_mask.close()
                finally:
                    mask.close()
            finally:
                scaled.close()
    finally:
        base.close()
    return results


def parse_angle_list(value: str) -> list[float]:
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if not parts:
        raise ValueError("Angle list cannot be empty.")
    return [float(item) for item in parts]


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    ry = np.array([[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(pitch), -np.sin(pitch)], [0.0, np.sin(pitch), np.cos(pitch)]], dtype=np.float32)
    rz = np.array([[np.cos(roll), -np.sin(roll), 0.0], [np.sin(roll), np.cos(roll), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ ry @ rx


def compute_gradient_magnitude(gray_uint8: np.ndarray, sobel_ksize: int, grad_norm_max: float) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for method 'perspective'. Install with: pip install opencv-python") from exc
    gray = gray_uint8.astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return np.clip(magnitude / float(grad_norm_max), 0.0, 1.0)


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
) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for method 'perspective'. Install with: pip install opencv-python") from exc

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
    cy = output_height / 2.0
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
        return cv2.resize(gradient, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    if crop_valid_region:
        try:
            x1, y1, x2, y2 = extract_max_fitting_bbox(valid_binary)
        except ValueError:
            ys, xs = np.nonzero(valid_binary)
            if xs.size == 0 or ys.size == 0:
                return cv2.resize(gradient, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
        cropped = warped[y1:y2, x1:x2]
        if cropped.size > 0:
            warped = cv2.resize(cropped, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
    return warped


def gradient_projection_to_embedding(
    gradient_projection: np.ndarray,
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
) -> list[float]:
    projection_uint8 = np.clip(np.round(gradient_projection * 255.0), 0, 255).astype(np.uint8)
    projection_image = Image.fromarray(projection_uint8, mode="L").convert("RGB")
    return compute_dino_embedding(
        image=projection_image,
        model=model,
        processor=processor,
        device=device,
        method="dinov2",
        patch_pooling="mean",
    )


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


def search_meilisearch(
    host: str,
    api_key: str | None,
    index_name: str,
    vector: list[float],
    limit: int = 10,
    filter_expression: str | None = None,
) -> dict[str, Any]:
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/search"
    payload = {
        "vector": vector,
        "limit": limit,
        "attributesToRetrieve": [
            "product_id",
            "projection_id",
            "yaw",
            "pitch",
            "roll",
            "name",
            "link",
            "image_url",
            "image_name",
            "image_path",
        ],
        "hybrid": {"semanticRatio": 1.0, "embedder": "default"},
        "showRankingScore": True,
    }
    if filter_expression:
        payload["filter"] = filter_expression
    try:
        return http_json("POST", endpoint, payload, api_key)
    except RuntimeError as exc:
        message = str(exc)
        if "Cannot find embedder with name `default`" not in message and "invalid_search_embedder" not in message:
            raise
        configure_index_embedder(host, api_key, index_name, len(vector))
        return http_json("POST", endpoint, payload, api_key)


def build_product_filter(product_ids: list[str]) -> str:
    if not product_ids:
        raise ValueError("Cannot build a product filter with an empty product id list.")
    escaped: list[str] = []
    for product_id in product_ids:
        text = str(product_id).replace("\\", "\\\\").replace('"', '\\"')
        escaped.append(f'"{text}"')
    return f"product_id IN [{','.join(escaped)}]"


def extract_unique_product_ids(hits: list[dict[str, Any]], limit: int) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        raw = hit.get("product_id") or hit.get("id")
        if raw is None:
            continue
        product_id = str(raw)
        if not product_id or product_id in seen:
            continue
        ids.append(product_id)
        seen.add(product_id)
        if len(ids) >= limit:
            break
    return ids


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


def search_documents_by_product_ids(
    host: str,
    api_key: str | None,
    index_name: str,
    product_ids: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if not product_ids:
        return []
    payload = {
        "q": "",
        "filter": build_product_filter(product_ids),
        "limit": limit,
        "attributesToRetrieve": [
            "id",
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
            "embedding",
        ],
        "showRankingScore": True,
    }
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/search"
    response = http_json("POST", endpoint, payload, api_key)
    if not isinstance(response, dict):
        return []
    raw_hits = response.get("hits", [])
    return raw_hits if isinstance(raw_hits, list) else []


def format_hybrid_match(hit: dict[str, Any], rank: int) -> dict[str, Any]:
    formatted = format_match(hit, rank)
    formatted["hsi_score"] = hit.get("hsi_score")
    formatted["dinov2_score"] = hit.get("dinov2_score")
    formatted["combined_score"] = hit.get("combined_score")
    formatted["dinov2_best_scale"] = hit.get("dinov2_best_scale")
    formatted["dinov2_best_rotation"] = hit.get("dinov2_best_rotation")
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
        "projection_id": hit.get("projection_id"),
        "yaw": hit.get("yaw"),
        "pitch": hit.get("pitch"),
        "roll": hit.get("roll"),
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


def format_faiss_match(hit: dict[str, Any], rank: int) -> dict[str, Any]:
    metadata = hit.get("metadata")
    merged = metadata if isinstance(metadata, dict) else {}
    image_value = hit.get("image") or merged.get("image")
    image_path = str(image_value) if image_value is not None else None
    name = merged.get("name") or merged.get("product_name") or merged.get("title")
    if not name:
        name = Path(image_path).stem if image_path else f"Match {rank}"
    score = hit.get("similarity")
    return {
        "rank": rank,
        "product_id": merged.get("product_id") or merged.get("id") or image_path,
        "name": name,
        "link": merged.get("link") or merged.get("product_url"),
        "image_url": merged.get("image_url"),
        "image_name": merged.get("image_name") or merged.get("filename") or (Path(image_path).name if image_path else None),
        "product_url": merged.get("product_url") or merged.get("link"),
        "preview_path": merged.get("preview_path"),
        "texture_url": merged.get("image_url"),
        "score": score,
        "ranking_score": score,
        "local_image": image_path,
        "metadata": merged,
    }


def merge_projection_hits(projection_hits: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    best_by_key: dict[str, dict[str, Any]] = {}
    for hit in projection_hits:
        key_parts = [
            str(hit.get("product_id") or ""),
            str(hit.get("link") or ""),
            str(hit.get("name") or ""),
        ]
        key = "|".join(key_parts)
        existing = best_by_key.get(key)
        current_score = hit.get("rankingScore")
        if current_score is None:
            current_score = hit.get("_rankingScore")
        if existing is None:
            best_by_key[key] = hit
            continue
        existing_score = existing.get("rankingScore")
        if existing_score is None:
            existing_score = existing.get("_rankingScore")
        current_score_value = float(current_score) if current_score is not None else float("-inf")
        existing_score_value = float(existing_score) if existing_score is not None else float("-inf")
        if current_score_value > existing_score_value:
            best_by_key[key] = hit

    sorted_hits = sorted(
        best_by_key.values(),
        key=lambda item: float(item.get("rankingScore") if item.get("rankingScore") is not None else item.get("_rankingScore") if item.get("_rankingScore") is not None else float("-inf")),
        reverse=True,
    )
    return sorted_hits[:limit]


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
        bbox_args = [args.bbox_x1, args.bbox_y1, args.bbox_x2, args.bbox_y2]
        if any(value is not None for value in bbox_args):
            if not all(value is not None for value in bbox_args):
                raise ValueError("When providing bbox coordinates, all of --bbox-x1/--bbox-y1/--bbox-x2/--bbox-y2 are required.")
            provided_bbox = (int(args.bbox_x1), int(args.bbox_y1), int(args.bbox_x2), int(args.bbox_y2))
            cropped, bbox = crop_image_from_bbox(image_array, provided_bbox)
        else:
            cropped, bbox = crop_image_from_mask(image_array, mask_array, margin=args.margin)
        cropped_image = Image.fromarray(cropped.astype(np.uint8))

        device = choose_device(args.device)
        method = args.method
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        embedding: list[float]
        matches: list[dict[str, Any]]
        hits: list[dict[str, Any]]

        if method == "gram":
            gram_extractor = VGGGramExtractor(layer_index=args.gram_layer_index).to(device)
            gram_extractor.eval()
            image_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
            embedding = compute_gram_embedding(
                image=cropped_image,
                gram_extractor=gram_extractor,
                image_transform=image_transform,
                device=device,
                lbp_points=args.lbp_points,
                lbp_radius=args.lbp_radius,
            )
        elif method == "hsi":
            embedding = compute_hsi_embedding(
                image=cropped_image,
                h_bins=args.h_bins,
                s_bins=args.s_bins,
                i_bins=args.i_bins,
            )
        elif method == "hsi_dinov2":
            if args.hsi_prefilter_limit <= 0:
                raise ValueError(f"--hsi-prefilter-limit must be > 0, got {args.hsi_prefilter_limit}.")
            if args.hsi_dinov2_size <= 0:
                raise ValueError(f"--hsi-dinov2-size must be > 0, got {args.hsi_dinov2_size}.")
            hsi_embedding = compute_hsi_embedding(
                image=cropped_image,
                h_bins=args.h_bins,
                s_bins=args.s_bins,
                i_bins=args.i_bins,
            )
            hsi_result = search_meilisearch(
                args.host,
                args.key,
                args.hsi_prefilter_index,
                hsi_embedding,
                limit=args.hsi_prefilter_limit,
            )
            hsi_hits = hsi_result.get("hits", []) if isinstance(hsi_result, dict) else []
            candidate_product_ids = extract_unique_product_ids(hsi_hits, args.hsi_prefilter_limit)
            if not candidate_product_ids:
                embedding = hsi_embedding
                hits = []
                matches = []
            else:
                image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME_DINOV2, token=hf_token)
                model = AutoModel.from_pretrained(MODEL_NAME_DINOV2, token=hf_token)
                model.to(device)
                model.eval()
                query_scales = parse_angle_list(args.hsi_dinov2_scales)
                query_rotations = parse_angle_list(args.hsi_dinov2_rotations)
                query_variants = compute_scaled_rotated_dino_queries(
                    image=cropped_image,
                    scales=query_scales,
                    rotations=query_rotations,
                    output_size=args.hsi_dinov2_size,
                    model=model,
                    processor=image_processor,
                    device=device,
                )
                if not query_variants:
                    raise RuntimeError("No transformed DINOv2 query embeddings were generated for hsi_dinov2.")
                baseline_query = next(
                    (
                        variant
                        for variant in query_variants
                        if abs(float(variant["scale"]) - 1.0) < 1e-6 and abs(float(variant["rotation"])) < 1e-6
                    ),
                    query_variants[0],
                )
                embedding = baseline_query["embedding"]
                hsi_scores = build_best_hsi_score_map(hsi_hits, args.hsi_prefilter_limit)
                candidate_docs = search_documents_by_product_ids(
                    host=args.host,
                    api_key=args.key,
                    index_name=args.index,
                    product_ids=candidate_product_ids,
                    limit=max(args.hsi_prefilter_limit, len(candidate_product_ids)),
                )
                best_by_product_id: dict[str, dict[str, Any]] = {}
                for doc in candidate_docs:
                    raw_product_id = doc.get("product_id") or doc.get("id")
                    if raw_product_id is None:
                        continue
                    product_id = str(raw_product_id)
                    if product_id not in hsi_scores:
                        continue

                    raw_doc_embedding = doc.get("embedding")
                    if not isinstance(raw_doc_embedding, list) or not raw_doc_embedding:
                        continue
                    doc_vec = np.asarray(raw_doc_embedding, dtype=np.float32).reshape(-1)
                    if doc_vec.size != len(embedding):
                        continue
                    doc_vec = l2_normalize(doc_vec.reshape(1, -1))[0]
                    best_dino_score = float("-inf")
                    best_scale = None
                    best_rotation = None
                    for variant in query_variants:
                        query_vec = variant["vector"]
                        dino_score = float(np.dot(query_vec, doc_vec))
                        if dino_score > best_dino_score:
                            best_dino_score = dino_score
                            best_scale = float(variant["scale"])
                            best_rotation = float(variant["rotation"])
                    if best_dino_score == float("-inf"):
                        continue
                    dino_score = best_dino_score
                    if dino_score < 0.0:
                        dino_score = 0.0
                    hsi_score = float(hsi_scores[product_id])
                    combined_score = hsi_score * dino_score

                    reranked = dict(doc)
                    reranked["hsi_score"] = hsi_score
                    reranked["dinov2_score"] = dino_score
                    reranked["dinov2_best_scale"] = best_scale
                    reranked["dinov2_best_rotation"] = best_rotation
                    reranked["combined_score"] = combined_score
                    reranked["score"] = combined_score
                    reranked["rankingScore"] = combined_score

                    existing = best_by_product_id.get(product_id)
                    if existing is None or float(existing["combined_score"]) < combined_score:
                        best_by_product_id[product_id] = reranked

                if candidate_docs and not best_by_product_id:
                    raise RuntimeError(
                        "HSI prefilter returned candidates, but no candidate had a usable DINOv2 embedding "
                        f"in index '{args.index}'."
                    )

                hits = sorted(
                    best_by_product_id.values(),
                    key=lambda item: float(item["combined_score"]),
                    reverse=True,
                )[: args.limit]
                matches = [format_hybrid_match(hit, index) for index, hit in enumerate(hits, start=1)]
        elif method == "perspective":
            image_processor = AutoImageProcessor.from_pretrained(args.perspective_model_name, token=hf_token)
            model = AutoModel.from_pretrained(args.perspective_model_name, token=hf_token)
            model.to(device)
            model.eval()

            gray = np.asarray(cropped_image.convert("L"), dtype=np.uint8)
            gradient = compute_gradient_magnitude(gray, args.sobel_ksize, args.grad_norm_max)
            yaw_angles = parse_angle_list(args.yaw_angles)
            pitch_angles = parse_angle_list(args.pitch_angles)

            collected_hits: list[dict[str, Any]] = []
            first_embedding: list[float] | None = None
            for yaw in yaw_angles:
                for pitch in pitch_angles:
                    projected = project_gradient(
                        gradient=gradient,
                        yaw=yaw,
                        pitch=pitch,
                        roll=args.perspective_roll,
                        output_width=args.perspective_output_width,
                        output_height=args.perspective_output_height,
                        focal_px=args.perspective_focal_length,
                        distance_px=args.perspective_camera_distance,
                        crop_valid_region=args.perspective_crop_valid_region,
                    )
                    projection_embedding = gradient_projection_to_embedding(projected, model, image_processor, device)
                    if first_embedding is None:
                        first_embedding = projection_embedding
                    projection_result = search_meilisearch(args.host, args.key, args.index, projection_embedding, limit=args.limit)
                    projection_hits = projection_result.get("hits", []) if isinstance(projection_result, dict) else []
                    collected_hits.extend(projection_hits)

            if first_embedding is None:
                raise RuntimeError("Perspective method failed to generate any query projection.")
            embedding = first_embedding
            hits = merge_projection_hits(collected_hits, args.limit)
            matches = [format_match(hit, index) for index, hit in enumerate(hits[: args.limit], start=1)]
        elif method == "dinov2_texture_faiss":
            try:
                from scripts.classification.texture_indexer_dinov2_texture_faiss import (
                    DINOv2TextureEmbedder,
                    DINOv2TextureSearchIndex,
                )
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Could not import DINOv2 FAISS texture search module. "
                    "Run from project root and ensure scripts/classification/texture_indexer_dinov2_texture_faiss.py exists."
                ) from exc

            faiss_embedder = DINOv2TextureEmbedder(
                embedding_mode=args.faiss_embedding_mode,
                input_size=args.faiss_input_size,
                device=device,
                batch_size=args.faiss_batch_size,
            )
            faiss_search = DINOv2TextureSearchIndex(index_root=args.faiss_index_root, embedder=faiss_embedder)
            faiss_search.load()
            query_embedding_np = faiss_embedder.embed_image(cropped_image)
            embedding = [float(v) for v in query_embedding_np.tolist()]
            faiss_results = faiss_search.search_by_embedding(query_embedding_np, top_k=args.limit)
            hits = faiss_results
            matches = [format_faiss_match(hit, index) for index, hit in enumerate(faiss_results, start=1)]
        else:
            model_name = MODEL_NAME_DINOV2 if method == "dinov2" else args.dinov3_model_name
            image_processor = AutoImageProcessor.from_pretrained(model_name, token=hf_token)
            model = AutoModel.from_pretrained(model_name, token=hf_token)
            model.to(device)
            model.eval()
            embedding = compute_dino_embedding(cropped_image, model, image_processor, device, method=method, patch_pooling=args.patch_pool)
        if method not in {"perspective", "hsi_dinov2", "dinov2_texture_faiss"}:
            result = search_meilisearch(args.host, args.key, args.index, embedding, limit=args.limit)
            hits = result.get("hits", []) if isinstance(result, dict) else []
            matches = [format_match(hit, index) for index, hit in enumerate(hits[: args.limit], start=1)]

        summary = {
            "image": str(image_path),
            "mask": str(mask_path),
            "bbox": bbox,
            "method": method,
            "embedding_dim": len(embedding),
            "matches": matches,
        }

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        output_json = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        try:
            sys.stdout.write(output_json)
        except UnicodeEncodeError:
            sys.stdout.buffer.write(output_json.encode("utf-8", errors="backslashreplace"))
        return 0
    finally:
        image.close()
        mask_image.close()


if __name__ == "__main__":
    raise SystemExit(main())
