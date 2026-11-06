#!/usr/bin/env python3
"""Use SigLIP2 to predict product attribute labels from rendered product images.

The script walks the downloaded `images/` folder, matches each filename to a product ID
using `ground_truth.json`, and then predicts the closest attribute values for:
- location
- application
- productCategory
- materialType
- color

It uses zero-shot matching between each image and the known vocabulary values from the
platform export. The output is a JSON object keyed by image filename.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


DEFAULT_EXPORT = "data/platform-products-production-v2-en.json"
DEFAULT_GROUND_TRUTH = "ground_truth.json"
DEFAULT_ATTRIBUTE_LABELS = "attribute_labels.json"
DEFAULT_IMAGES_DIR = "images"
DEFAULT_OUTPUT = "image_predictions.json"
MODEL_NAME = "google/siglip2-base-patch16-224"


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
    }

    for record in records:
        for field in catalog:
            values = record.get(field, [])
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                continue
            for value in values:
                text = normalize_value(value)
                if text and text not in catalog[field]:
                    catalog[field].append(text)

    for field, values in catalog.items():
        catalog[field] = sorted(values)
    return catalog


def load_attribute_catalog(attribute_labels_path: Path, product_export_path: Path) -> dict[str, list[str]]:
    if attribute_labels_path.exists():
        labels = load_json(attribute_labels_path)
        if not isinstance(labels, dict):
            raise ValueError(f"Expected a dictionary in {attribute_labels_path}")
        catalog: dict[str, list[str]] = {}
        for field in ["location", "application", "productCategory", "materialType", "color"]:
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


def build_prompt(field_name: str, value: str) -> str:
    field_medium = {
        "location": "room scene",
        "application": "product use case",
        "productCategory": "product category",
        "materialType": "material",
        "color": "color",
    }.get(field_name, "product")
    return f"{field_name}: {value}; {field_medium}."


def extract_top_predictions(scores: list[float], labels: list[str], k: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(
        zip(labels, scores),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"label": label, "score": round(float(score), 6)}
        for label, score in ranked[:k]
    ]


def predict_attributes_for_image(
    image_path: Path,
    processor: AutoProcessor,
    model: AutoModel,
    catalog: dict[str, list[str]],
    device: str,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    predictions: dict[str, Any] = {}

    for field_name, values in catalog.items():
        prompts = [build_prompt(field_name, value) for value in values]
        if not prompts:
            continue

        inputs = processor(
            images=[image],
            text=prompts,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image[0]

        scores = logits.cpu().tolist()
        predictions[field_name] = {
            "top": extract_top_predictions(scores, values, k=3),
            "best": values[int(torch.argmax(logits).item())],
            "best_score": round(float(torch.max(logits).item()), 6),
        }

    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict product attributes from gallery images using SigLIP2.")
    parser.add_argument("--product-export", type=Path, default=Path(DEFAULT_EXPORT), help="Path to product JSON export")
    parser.add_argument("--ground-truth", type=Path, default=Path(DEFAULT_GROUND_TRUTH), help="Path to filename-to-product-id mapping")
    parser.add_argument("--attribute-labels", type=Path, default=Path(DEFAULT_ATTRIBUTE_LABELS), help="JSON file of attribute labels to use for classification")
    parser.add_argument("--images-dir", type=Path, default=Path(DEFAULT_IMAGES_DIR), help="Directory with downloaded product images")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT), help="Where to write JSON predictions")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N image files (0 = all)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device used for inference")
    args = parser.parse_args()

    catalog = load_attribute_catalog(args.attribute_labels, args.product_export)

    ground_truth = load_json(args.ground_truth)
    if not isinstance(ground_truth, dict):
        raise ValueError(f"Expected a dictionary in {args.ground_truth}")

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

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    predictions: dict[str, Any] = {}
    for image_path in files:
        product_id = None
        for filename, pid in ground_truth.items():
            if filename == image_path.name:
                product_id = str(pid)
                break

        image_result: dict[str, Any] = {
            "product_id": product_id,
            "predicted": predict_attributes_for_image(image_path, processor, model, catalog, device),
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
