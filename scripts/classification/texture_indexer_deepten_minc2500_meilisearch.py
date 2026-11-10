#!/usr/bin/env python3
"""Index texture embeddings using DeepTEN (MINC-2500) into Meilisearch.

This script loads products from a Lightbeans export, resolves each product texture file,
computes DeepTEN embeddings from the pretrained MINC-2500 model, and uploads documents
with vectors to Meilisearch for semantic retrieval.
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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

DEFAULT_PRODUCT_EXPORT = "texture/lightbeans_main_page_textures.json"
DEFAULT_TEXTURE_DIR = "texture/main_page"
DEFAULT_MODEL_NAME = "deepten_resnet50_minc"
MEILI_INDEX = "lightbeans-texture2-deepten-minc2500"


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


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")
    return device_arg


def load_deepten_model(model_name: str, pretrained: bool, model_root: Path, device: str):
    try:
        from encoding.models.deepten import get_deepten_resnet50_minc
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch-Encoding is required for DeepTEN indexing. "
            "Install it with: pip install git+https://github.com/zhanghang1989/PyTorch-Encoding.git"
        ) from exc

    try:
        if model_name == DEFAULT_MODEL_NAME:
            model = get_deepten_resnet50_minc(pretrained=pretrained, root=str(model_root.expanduser()))
        else:
            raise RuntimeError(
                f"Unsupported model '{model_name}' in this environment. "
                f"Use '{DEFAULT_MODEL_NAME}'."
            )
    except Exception as exc:
        raise RuntimeError(f"Could not load DeepTEN model '{model_name}': {exc}") from exc

    head_layers = list(model.head.children())
    if len(head_layers) < 2:
        raise RuntimeError("Unexpected DeepTEN head structure; cannot remove final classifier layer.")
    model.head = torch.nn.Sequential(*head_layers[:-1])
    model.to(device)
    model.eval()
    return model


def compute_deepten_embedding(image_path: Path, model: torch.nn.Module, transform: transforms.Compose, device: str) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    try:
        batch = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model(batch)
            if feature.ndim != 2:
                feature = feature.reshape(feature.shape[0], -1)
            if feature.shape[1] != 4096:
                raise RuntimeError(f"Expected DeepTEN pre-classifier embedding dimension 4096, got {feature.shape[1]}.")
            feature = F.normalize(feature, p=2, dim=1)
        vector = feature.detach().cpu().numpy().astype(np.float32)[0]
        return [float(v) for v in vector.tolist()]
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


def upload_documents(host: str, api_key: str | None, index_name: str, documents: list[dict[str, Any]], batch_size: int = 50) -> None:
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


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    env_values = load_env(project_root / ".env")
    parser = argparse.ArgumentParser(description="Index DeepTEN MINC-2500 texture vectors into Meilisearch.")
    parser.add_argument("--product-export", type=Path, default=Path(DEFAULT_PRODUCT_EXPORT), help="Product export JSON file.")
    parser.add_argument("--textures-dir", type=Path, default=Path(DEFAULT_TEXTURE_DIR), help="Directory containing downloaded texture files.")
    parser.add_argument("--mapping-json", type=Path, default=None, help="Optional file-to-product mapping JSON for legacy texture exports.")
    parser.add_argument("--host", default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700", help="Meilisearch host.")
    parser.add_argument("--key", default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"), help="Meilisearch API key.")
    parser.add_argument("--index", default=env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or MEILI_INDEX, help="Target Meilisearch index name.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Encoding model name (default: deepten_resnet50_minc).")
    parser.add_argument("--model-root", type=Path, default=Path("~/.encoding/models"), help="Directory for cached Encoding model weights.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for embedding inference.")
    parser.add_argument("--image-size", type=int, default=256, help="Resize shorter side before center crop.")
    parser.add_argument("--crop-size", type=int, default=224, help="Center crop size used for DeepTEN input.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N products (0 = all).")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of documents per Meilisearch upload batch.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load pretrained MINC-2500 weights (not recommended for retrieval quality).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.product_export.exists():
        raise FileNotFoundError(f"Product export not found: {args.product_export}")
    if not args.textures_dir.exists():
        raise FileNotFoundError(f"Textures directory not found: {args.textures_dir}")
    if args.image_size <= 0 or args.crop_size <= 0:
        raise ValueError("--image-size and --crop-size must be > 0.")

    products = load_product_export(args.product_export)
    if args.limit > 0:
        products = products[: args.limit]

    device = choose_device(args.device)
    model = load_deepten_model(
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
        model_root=args.model_root,
        device=device,
    )
    image_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

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

        vector = compute_deepten_embedding(texture_path, model, image_transform, device)
        documents.append(build_document(record, texture_path, vector))

    vector_dimensions = len(documents[0]["embedding"]) if documents else None
    ensure_index(args.host, args.key, args.index, vector_dimensions=vector_dimensions)
    upload_documents(args.host, args.key, args.index, documents, batch_size=args.batch_size)
    print(f"Indexed {len(documents)} DeepTEN vectors in Meilisearch index '{args.index}'.")
    print(f"Skipped {skipped} products without a usable texture file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
