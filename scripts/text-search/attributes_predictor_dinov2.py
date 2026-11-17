#!/usr/bin/env python3
"""Use DINOv2 image embeddings + sentence-transformers text embeddings to predict product attribute labels.

This mirrors the SigLIP2 script but relies on a vision-only DINOv2 model for image embeddings and a
SentenceTransformers model for label embeddings. The image and label embeddings are compared using cosine
similarity, and the closest label per attribute is selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel


DEFAULT_EXPORT = "texture/lightbeans_main_page_textures.json"
DEFAULT_GROUND_TRUTH = "texture/lightbeans_main_page_textures.json"
DEFAULT_ATTRIBUTE_LABELS = "attribute_labels.json"
DEFAULT_IMAGES_DIR = "texture/main_page"
DEFAULT_OUTPUT = "results/DINOv2_image_predictions.json"
MODEL_NAME = "facebook/dinov2-base"
TEXT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def normalize_value(value: Any) -> str:
    return str(value).strip()


def build_attribute_catalog(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    catalog: dict[str, list[str]] = {
        "location": [],
        "application": [],
        "productCategory": [],
        "materialType": [],
        "color": [],
        "name": [],
        "link": [],
    }

    for record in records:
        if not isinstance(record, dict):
            continue

        legacy_fields = [
            ("location", record.get("location")),
            ("application", record.get("application")),
            ("productCategory", record.get("productCategory")),
            ("materialType", record.get("materialType")),
            ("color", record.get("color")),
        ]
        for field_name, values in legacy_fields:
            if values is None:
                continue
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                continue
            for value in values:
                text = normalize_value(value)
                if text and text not in catalog[field_name]:
                    catalog[field_name].append(text)

        fallback_fields = [
            ("name", record.get("name")),
            ("link", record.get("link")),
        ]
        for field_name, value in fallback_fields:
            if value is None:
                continue
            text = normalize_value(value)
            if text and text not in catalog[field_name]:
                catalog[field_name].append(text)

    for field, values in catalog.items():
        catalog[field] = sorted(values)
    return catalog


def load_attribute_catalog(attribute_labels_path: Path, product_export_path: Path) -> dict[str, list[str]]:
    if attribute_labels_path.exists():
        labels = load_json(attribute_labels_path)
        if not isinstance(labels, dict):
            raise ValueError(f"Expected a dictionary in {attribute_labels_path}")
        catalog: dict[str, list[str]] = {}
        for field in ["location", "application", "productCategory", "materialType", "color", "name", "link"]:
            values = labels.get(field, [])
            if isinstance(values, list):
                catalog[field] = [str(value).strip() for value in values if str(value).strip()]
            else:
                catalog[field] = []
        return catalog

    records = load_json(product_export_path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {product_export_path}")
    return build_attribute_catalog(records)


def build_text_label(field_name: str, value: str) -> str:
    return f"{field_name}: {value}"


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def image_embedding_from_model(
    image: Image.Image,
    model: AutoModel,
    image_processor: AutoImageProcessor,
    device: str,
) -> np.ndarray:
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        embedding = outputs.pooler_output[0].cpu().numpy()
    elif hasattr(outputs, "last_hidden_state"):
        embedding = outputs.last_hidden_state[0, 0].cpu().numpy()
    else:
        raise RuntimeError("DINOv2 output does not expose a usable pooled embedding.")

    return l2_normalize(np.asarray(embedding, dtype=np.float32))


def extract_top_predictions(scores: np.ndarray, labels: list[str], k: int = 3) -> list[dict[str, Any]]:
    ranked_indices = np.argsort(scores)[::-1][:k]
    results = []
    for idx in ranked_indices:
        label = labels[int(idx)]
        score = float(scores[int(idx)])
        results.append({"label": label, "score": round(score, 6)})
    return results


def predict_attributes_for_image(
    image_path: Path,
    image_model: AutoModel,
    image_processor: AutoImageProcessor,
    text_model: SentenceTransformer,
    catalog: dict[str, list[str]],
    device: str,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    image_embed = image_embedding_from_model(image, image_model, image_processor, device)
    predictions: dict[str, Any] = {}

    for field_name, values in catalog.items():
        if not values:
            continue

        prompts = [build_text_label(field_name, value) for value in values]
        text_embeds = text_model.encode(prompts, convert_to_numpy=True, normalize_embeddings=True)
        text_embeds = np.asarray(text_embeds, dtype=np.float32)
        scores = image_embed @ text_embeds.T
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        best_idx = int(np.argmax(scores))
        predictions[field_name] = {
            "top": extract_top_predictions(scores, values, k=3),
            "best": values[best_idx],
            "best_score": round(float(scores[best_idx]), 6),
        }

    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict product attribute labels from gallery images using DINOv2 image embeddings and a text encoder.")
    parser.add_argument("--product-export", type=Path, default=Path(DEFAULT_EXPORT), help="Path to product JSON export")
    parser.add_argument("--ground-truth", type=Path, default=Path(DEFAULT_GROUND_TRUTH), help="Path to filename-to-product-id mapping")
    parser.add_argument("--attribute-labels", type=Path, default=Path(DEFAULT_ATTRIBUTE_LABELS), help="JSON file with attribute labels used for classification")
    parser.add_argument("--images-dir", type=Path, default=Path(DEFAULT_IMAGES_DIR), help="Directory with downloaded product images")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT), help="Where to write JSON predictions")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N images (0 = all)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device used for inference")
    args = parser.parse_args()

    catalog = load_attribute_catalog(args.attribute_labels, args.product_export)

    ground_truth_raw = load_json(args.ground_truth)
    if isinstance(ground_truth_raw, dict):
        ground_truth = ground_truth_raw
    elif isinstance(ground_truth_raw, list):
        ground_truth = {}
        for idx, record in enumerate(ground_truth_raw, start=1):
            if not isinstance(record, dict):
                continue
            image_name = record.get("image_name") or record.get("imageName")
            if isinstance(image_name, str) and image_name.strip():
                ground_truth[image_name.strip()] = idx
            image_url = record.get("image_url") or record.get("imageUrl")
            if isinstance(image_url, str) and image_url.strip():
                ground_truth[Path(image_url.strip()).name] = idx
    else:
        raise ValueError(f"Expected a dictionary or list in {args.ground_truth}")

    images_dir = args.images_dir
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    files = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if args.limit > 0:
        files = files[: args.limit]

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    image_model = AutoModel.from_pretrained(MODEL_NAME)
    image_model.to(device)
    image_model.eval()

    text_model = SentenceTransformer(TEXT_MODEL_NAME)
    predictions: dict[str, Any] = {}

    for image_path in files:
        product_id = None
        matched_key = ground_truth.get(image_path.name)
        if matched_key is not None:
            product_id = str(matched_key)
        else:
            for filename, pid in ground_truth.items():
                if filename == image_path.name:
                    product_id = str(pid)
                    break

        image_result: dict[str, Any] = {
            "product_id": product_id,
            "image_name": image_path.name,
            "predicted": predict_attributes_for_image(image_path, image_model, image_processor, text_model, catalog, device),
        }
        predictions[image_path.name] = image_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(predictions, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Processed {len(predictions)} images and wrote predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
