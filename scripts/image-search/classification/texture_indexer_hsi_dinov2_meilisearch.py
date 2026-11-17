#!/usr/bin/env python3
"""Index hybrid HSI + DINOv2 view embeddings into Meilisearch."""

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

MODEL_NAME = "facebook/dinov2-small"
DEFAULT_FILTER_METADATA_EXPORT = "data/platform-products-production-v2-en.json"
DEFAULT_TEXTURE_DIR = "texture/views"
DEFAULT_MAIN_PAGE_DIR = "texture/main"
MEILI_INDEX = "lightbeans-texture-hsi-dinov2-views-15"
EXPECTED_VIEWS_PER_FOLDER = 15

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
        token = str(api_key).strip()
        if token:
            headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    data = None if payload is None else json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
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


def load_filter_metadata_export(path: Path) -> list[dict[str, Any]]:
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


def collect_texture_images(texture_dir: Path) -> list[Path]:
    allowed_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    images: list[Path] = []
    for path in sorted(texture_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_ext:
            continue
        file_name = path.name.lower()
        if file_name == "contact_sheet.jpg" or path.stem.lower().startswith("contact_sheet"):
            continue
        images.append(path)
    return images


def collect_view_folders(texture_dir: Path) -> list[tuple[Path, list[Path]]]:
    folder_to_images: dict[Path, list[Path]] = {}
    for image_path in collect_texture_images(texture_dir):
        folder_to_images.setdefault(image_path.parent, []).append(image_path)
    folders = sorted(folder_to_images.keys())
    return [(folder, sorted(folder_to_images[folder])) for folder in folders]


def build_main_page_image_lookup(main_page_dir: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for image_path in collect_texture_images(main_page_dir):
        lookup.setdefault(normalize_stem(image_path.name), image_path)
    return lookup


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
    tid = record.get("tid")
    if isinstance(tid, str) and tid.strip():
        candidates.append(normalize_stem(tid))
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


def resolve_main_page_image_for_record(
    record: dict[str, Any],
    main_page_lookup: dict[str, Path],
    folder_stem: str,
) -> Path | None:
    candidates: list[str] = [folder_stem]
    tid = record.get("tid")
    if isinstance(tid, str) and tid.strip():
        candidates.insert(0, normalize_stem(tid))
    # image_name = record.get("image_name")
    # if isinstance(image_name, str) and image_name.strip():
    #     candidates.append(normalize_stem(image_name))
    # image_url = record.get("image_url")
    # if isinstance(image_url, str) and image_url.strip():
    #     filename = image_url.rstrip("/").split("/")[-1]
    #     if filename:
    #         candidates.append(normalize_stem(filename))
    # preview_path = normalize_preview_path(record.get("previewPath"))
    # if preview_path:
    #     candidates.append(normalize_stem(preview_path))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        path = main_page_lookup.get(candidate)
        if path is not None and path.exists():
            return path
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
        "searchableAttributes": [
            "brandName",
            "productName",
            "location",
            "application",
            "productCategory",
            "materialType",
            "color",
        ],
        "filterableAttributes": [
            "brandName",
            "productName",
            "location",
            "application",
            "productCategory",
            "materialType",
            "color",
        ],
        "sortableAttributes": [],
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


def compute_dinov2_embeddings(
    image_paths: list[Path],
    model: AutoModel,
    processor: AutoImageProcessor,
    device: str,
    batch_size: int,
) -> list[list[float]]:
    model.eval()
    output: list[list[float]] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images: list[Image.Image] = []
        for image_path in batch_paths:
            images.append(Image.open(image_path).convert("RGB"))
        try:
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state"):
                    embedding = outputs.last_hidden_state[:, 0, :]
                else:
                    raise RuntimeError("DINOv2 output does not expose a usable embedding.")
            embedding = embedding.detach().cpu().numpy().astype(np.float32)
            embedding = l2_normalize(embedding)
            output.extend([[float(v) for v in row.tolist()] for row in embedding])
        finally:
            for image in images:
                image.close()
    return output


def average_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("Cannot average an empty vector list.")
    matrix = np.asarray(vectors, dtype=np.float32)
    mean_vector = matrix.mean(axis=0)
    normalized = l2_normalize(np.asarray([mean_vector], dtype=np.float32))[0]
    return [float(v) for v in normalized.tolist()]


def build_product_url(brand_slug: Any, product_slug: Any) -> str | None:
    if not isinstance(brand_slug, str) or not isinstance(product_slug, str):
        return None
    brand = brand_slug.strip().strip("/")
    product = product_slug.strip().strip("/")
    if not brand or not product:
        return None
    return f"https://lightbeans.com/en/texture/{brand}/{product}"


def build_document(
    folder_path: Path,
    hsi_image_path: Path,
    view_paths: list[Path],
    hsi_vector: list[float],
    dinov2_vectors: list[list[float]],
    record: dict[str, Any],
) -> dict[str, Any]:
    folder_id = sanitize_meilisearch_id(folder_path.name)
    product_id = record.get("id") or record.get("image_name") or record.get("link") or folder_id
    product_id = sanitize_meilisearch_id(product_id)
    link = record.get("link") or build_product_url(record.get("brandSlug"), record.get("productSlug"))
    image_url = record.get("image_url")
    if not image_url and isinstance(record.get("previewPath"), str):
        preview_path = normalize_preview_path(record.get("previewPath"))
        if preview_path:
            image_url = f"https://cdn.lightbeans.com/textures/{preview_path.lstrip('/')}"

    return {
        "id": folder_id,
        "folder_name": folder_path.name,
        "folder_path": folder_path.as_posix(),
        "product_id": product_id,
        "name": record.get("name"),
        "link": link,
        "image_url": image_url or "",
        "source_image_name": record.get("image_name") or folder_path.name,
        "image_names": [path.name for path in view_paths],
        "image_paths": [path.as_posix() for path in view_paths],
        "hsi_image_path": hsi_image_path.as_posix(),
        "view_count": len(view_paths),
        "brandName": record.get("brandName"),
        "productName": record.get("productName"),
        "location": as_list(record.get("location")),
        "application": as_list(record.get("application")),
        "productCategory": as_list(record.get("productCategory")),
        "materialType": as_list(record.get("materialType")),
        "color": as_list(record.get("color")),
        "locale": record.get("locale"),
        "dinov2_embedding_vectors": dinov2_vectors,
        "_vectors": {"default": hsi_vector},
    }


def upload_documents(host: str, api_key: str | None, index_name: str, documents: list[dict[str, Any]]) -> None:
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/documents"
    response = http_json("POST", endpoint, documents, api_key, retries=5, retry_delay_seconds=2.0)
    wait_for_task(host, api_key, extract_task_uid(response, "upload-documents"))


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    env_values = load_env(project_root / ".env")
    parser = argparse.ArgumentParser(description="Index hybrid HSI + DINOv2 texture view embeddings into Meilisearch.")
    parser.add_argument("--filter-metadata-export", type=Path, default=Path(DEFAULT_FILTER_METADATA_EXPORT), help="Filter metadata JSON file.")
    parser.add_argument("--textures-dir", type=Path, default=Path(DEFAULT_TEXTURE_DIR), help="Directory containing texture view folders.")
    parser.add_argument("--main-page-dir", type=Path, default=Path(DEFAULT_MAIN_PAGE_DIR), help="Directory containing main-page texture images for HSI.")
    parser.add_argument("--host", default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700", help="Meilisearch host.")
    parser.add_argument("--key", default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"), help="Meilisearch API key.")
    parser.add_argument("--index", default=env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or MEILI_INDEX, help="Target Meilisearch index name.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N metadata records (0 = all).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for DINOv2 inference.")
    parser.add_argument("--expected-views-per-folder", type=int, default=EXPECTED_VIEWS_PER_FOLDER, help="Expected number of view images per folder.")
    parser.add_argument("--batch-size", type=int, default=16, help="DINOv2 inference batch size.")
    parser.add_argument("--upload-batch-size", type=int, default=64, help="Documents per upload batch.")
    parser.add_argument("--h-bins", type=int, default=36, help="Hue histogram bins.")
    parser.add_argument("--s-bins", type=int, default=16, help="Saturation histogram bins.")
    parser.add_argument("--i-bins", type=int, default=16, help="Intensity histogram bins.")
    parser.add_argument("--trace", dest="trace", action="store_true", help="Print progression traces while indexing.")
    parser.add_argument("--no-trace", dest="trace", action="store_false", help="Disable progression traces while indexing.")
    parser.set_defaults(trace=True)
    parser.add_argument("--trace-every", type=int, default=1, help="Emit progress every N metadata records.")
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
    if not args.main_page_dir.exists():
        raise FileNotFoundError(f"Main-page directory not found: {args.main_page_dir}")
    if args.expected_views_per_folder <= 0:
        raise ValueError("--expected-views-per-folder must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.upload_batch_size <= 0:
        raise ValueError("--upload-batch-size must be > 0.")
    if args.trace_every <= 0:
        raise ValueError("--trace-every must be > 0.")

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
    view_folders = collect_view_folders(args.textures_dir)
    if not view_folders:
        raise RuntimeError(f"No supported images found in {args.textures_dir}.")
    folder_lookup = build_view_folder_lookup(view_folders)
    main_page_lookup = build_main_page_image_lookup(args.main_page_dir)
    if not main_page_lookup:
        raise RuntimeError(f"No supported images found in {args.main_page_dir}.")

    if args.trace:
        print(
            "[trace] Starting HSI+DINOv2 indexing "
            f"(metadata={total_records}, folders={len(view_folders)}, expected_views={args.expected_views_per_folder})."
        )

    documents: list[dict[str, Any]] = []
    skipped_no_folder = 0
    skipped_no_main_page = 0
    skipped_invalid_views = 0
    skipped_duplicates = 0
    processed_folder_ids: set[str] = set()

    for record_index, metadata_record in enumerate(metadata_records, start=1):
        resolved_record = dict(metadata_record)

        resolved = resolve_view_folder_for_record(resolved_record, folder_lookup)
        if resolved is None:
            skipped_no_folder += 1
            if args.trace:
                print(f"[trace] Skipped {record_index}/{total_records}: no matching view folder.")
            continue
        folder_path, view_paths = resolved
        folder_stem = normalize_stem(folder_path.name)
        main_page_image = resolve_main_page_image_for_record(resolved_record, main_page_lookup, folder_stem)
        if main_page_image is None:
            skipped_no_main_page += 1
            if args.trace:
                print(f"[trace] Skipped {record_index}/{total_records}: no matching main-page image for folder '{folder_path.name}'.")
            continue

        folder_id = sanitize_meilisearch_id(folder_path.name)
        if folder_id in processed_folder_ids:
            skipped_duplicates += 1
            if args.trace:
                print(f"[trace] Skipped {record_index}/{total_records}: duplicate folder match '{folder_path.name}'.")
            continue
        processed_folder_ids.add(folder_id)

        if len(view_paths) != args.expected_views_per_folder:
            skipped_invalid_views += 1
            if args.trace:
                print(
                    f"[trace] Skipped {record_index}/{total_records}: folder '{folder_path.name}' "
                    f"expected {args.expected_views_per_folder} views but found {len(view_paths)}."
                )
            continue

        hsi_vector = compute_hsi_histogram(main_page_image, args.h_bins, args.s_bins, args.i_bins)

        dinov2_vectors = compute_dinov2_embeddings(view_paths, image_model, image_processor, device, args.batch_size)

        documents.append(build_document(folder_path, main_page_image, view_paths, hsi_vector, dinov2_vectors, resolved_record))

        if args.trace and (record_index == 1 or record_index % args.trace_every == 0 or record_index == total_records):
            print(
                f"[trace] Progress {record_index}/{total_records}: indexed={len(documents)}, "
                f"skipped_no_folder={skipped_no_folder}, "
                f"skipped_no_main_page={skipped_no_main_page}, skipped_invalid_views={skipped_invalid_views}, "
                f"skipped_duplicates={skipped_duplicates}."
            )

    if not documents:
        raise RuntimeError("No documents generated for indexing.")

    default_vector = documents[0].get("_vectors", {}).get("default")
    if not isinstance(default_vector, list) or not default_vector:
        raise RuntimeError("First document is missing a valid _vectors.default embedding.")
    vector_dimensions = len(default_vector)
    ensure_index(args.host, args.key, args.index, vector_dimensions=vector_dimensions)
    for start in range(0, len(documents), args.upload_batch_size):
        batch = documents[start : start + args.upload_batch_size]
        upload_documents(args.host, args.key, args.index, batch)
        if args.trace:
            end = min(start + len(batch), len(documents))
            print(f"[trace] Uploaded {end}/{len(documents)} documents.")

    print(f"Indexed {len(documents)} documents in Meilisearch index '{args.index}'.")
    print(f"Skipped {skipped_no_folder} metadata records without folder match.")
    print(f"Skipped {skipped_no_main_page} records without matching main-page image.")
    print(f"Skipped {skipped_invalid_views} folders with invalid view count.")
    print(f"Skipped {skipped_duplicates} duplicate folder matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
