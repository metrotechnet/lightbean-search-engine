#!/usr/bin/env python3
"""Index texture vectors (LBP + VGG Gram) into Meilisearch.

Inspired by:
https://github.com/real-ojaswi/texture-analysis-of-images
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

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19

DEFAULT_PRODUCT_EXPORT = "texture/lightbeans_main_page_textures.json"
DEFAULT_TEXTURE_DIR = "texture/main_page"
MEILI_INDEX = "lightbeans-texture2-texture-analysis"


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
    payload = {"uid": index_name, "primaryKey": "product_id"}
    try:
        response = http_json("POST", endpoint, payload, api_key)
        wait_for_task(host, api_key, extract_task_uid(response, "create-index"))
    except RuntimeError as exc:
        message = str(exc).lower()
        if "already exists" not in message and "index_already_exists" not in message:
            raise

    settings: dict[str, Any] = {
        "searchableAttributes": ["name", "link", "image_url", "image_name"],
        "filterableAttributes": ["product_id", "name", "link", "image_url", "image_name"],
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
        theta = 2.0 * math.pi * bit / points
        dy = int(round(-radius * math.sin(theta)))
        dx = int(round(radius * math.cos(theta)))
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
        fmap = self.features(x)  # [B,C,H,W]
        b, c, h, w = fmap.shape
        fm = fmap.view(b, c, h * w)
        gram = torch.bmm(fm, fm.transpose(1, 2)) / float(c * h * w)
        return gram


def gram_upper_triangle(gram: np.ndarray) -> np.ndarray:
    c, _ = gram.shape
    tri = np.triu_indices(c)
    return gram[tri].astype(np.float32)


def compute_texture_vector(
    image_path: Path,
    gram_extractor: VGGGramExtractor,
    image_transform: transforms.Compose,
    device: str,
    lbp_points: int,
    lbp_radius: int,
) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    try:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        lbp_hist = compute_lbp_histogram(gray, points=lbp_points, radius=lbp_radius)

        tensor = image_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            gram = gram_extractor(tensor)[0].detach().cpu().numpy()
        gram_vec = gram_upper_triangle(gram)

        feature = np.concatenate([lbp_hist, gram_vec], axis=0).astype(np.float32)
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


def build_document(product: dict[str, Any], image_path: Path, vector: list[float]) -> dict[str, Any]:
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
        "embedding": vector,
        "_vectors": {"default": vector},
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
        wait_for_task(host, api_key, extract_task_uid(response, "upload-documents"))


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
        ],
        "hybrid": {"semanticRatio": 1.0, "embedder": "default"},
        "showRankingScore": True,
    }
    return http_json("POST", endpoint, payload, api_key)


def format_match_for_viewer(hit: dict[str, Any], rank: int) -> dict[str, Any]:
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
    product_url = hit.get("link") or hit.get("product_url") or hit.get("productUrl")
    image_url = hit.get("image_url") or hit.get("texture_url") or hit.get("preview_path")
    return {
        "rank": rank,
        "product_id": hit.get("product_id") or hit.get("id"),
        "name": hit.get("name") or hit.get("product_name") or hit.get("productName"),
        "link": product_url,
        "image_url": image_url,
        "image_name": hit.get("image_name") or hit.get("imageName"),
        "product_url": product_url,
        "preview_path": hit.get("preview_path") or hit.get("previewPath"),
        "texture_url": image_url,
        "score": score,
        "ranking_score": hit.get("rankingScore") or hit.get("_rankingScore"),
    }


def search_by_image_for_viewer(
    host: str,
    api_key: str | None,
    index_name: str,
    image_path: Path,
    gram_extractor: VGGGramExtractor,
    image_transform: transforms.Compose,
    device: str,
    lbp_points: int,
    lbp_radius: int,
    limit: int = 10,
) -> dict[str, Any]:
    vector = compute_texture_vector(
        image_path=image_path,
        gram_extractor=gram_extractor,
        image_transform=image_transform,
        device=device,
        lbp_points=lbp_points,
        lbp_radius=lbp_radius,
    )
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
    parser = argparse.ArgumentParser(description="Index texture vectors (LBP + VGG Gram) into Meilisearch.")
    parser.add_argument("--product-export", type=Path, default=Path(DEFAULT_PRODUCT_EXPORT), help="Product export JSON file.")
    parser.add_argument("--textures-dir", type=Path, default=Path(DEFAULT_TEXTURE_DIR), help="Directory containing downloaded texture files.")
    parser.add_argument("--mapping-json", type=Path, default=None, help="Optional file-to-product mapping JSON for legacy texture exports.")
    parser.add_argument("--host", default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700", help="Meilisearch host.")
    parser.add_argument("--key", default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"), help="Meilisearch API key.")
    parser.add_argument("--index", default=env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or MEILI_INDEX, help="Target Meilisearch index name.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N products (0 = all).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for extraction.")
    parser.add_argument("--lbp-points", type=int, default=8, help="LBP neighborhood points.")
    parser.add_argument("--lbp-radius", type=int, default=1, help="LBP radius.")
    parser.add_argument("--gram-layer-index", type=int, default=8, help="VGG19 feature layer index used to compute Gram matrix.")
    parser.add_argument("--upload-batch-size", type=int, default=100, help="Number of documents per Meilisearch upload batch.")
    parser.add_argument("--query-image", type=Path, help="Optional query image to run viewer-compatible vector search.")
    parser.add_argument("--search-limit", type=int, default=10, help="Number of hits to return for --query-image.")
    parser.add_argument("--search-output", type=Path, help="Optional output JSON path for viewer-compatible search payload.")
    return parser.parse_args()


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

    device = choose_device(args.device)
    gram_extractor = VGGGramExtractor(layer_index=args.gram_layer_index).to(device)
    gram_extractor.eval()
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    products = load_product_export(args.product_export)
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

        vector = compute_texture_vector(
            image_path=texture_path,
            gram_extractor=gram_extractor,
            image_transform=image_transform,
            device=device,
            lbp_points=args.lbp_points,
            lbp_radius=args.lbp_radius,
        )
        documents.append(build_document(record, texture_path, vector))

    vector_dimensions = len(documents[0]["embedding"]) if documents else None
    ensure_index(args.host, args.key, args.index, vector_dimensions=vector_dimensions)
    upload_documents(args.host, args.key, args.index, documents, batch_size=args.upload_batch_size)
    print(f"Indexed {len(documents)} texture-analysis vectors in Meilisearch index '{args.index}'.")
    print(f"Skipped {skipped} products without a usable texture file.")

    if args.query_image:
        if not args.query_image.exists():
            raise FileNotFoundError(f"Query image not found: {args.query_image}")
        search_payload = search_by_image_for_viewer(
            host=args.host,
            api_key=args.key,
            index_name=args.index,
            image_path=args.query_image,
            gram_extractor=gram_extractor,
            image_transform=image_transform,
            device=device,
            lbp_points=args.lbp_points,
            lbp_radius=args.lbp_radius,
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
