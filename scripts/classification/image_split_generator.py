#!/usr/bin/env python3
"""Split images into train/test sets using first product occurrence.

Rules:
1) train: first image occurrence for each product_id
2) test: all other image occurrences for the same product_id
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def collect_images(images_dir: Path) -> list[str]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return sorted(
        path.name
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )


def split_by_first_product_occurrence(image_names: list[str], ground_truth: dict[str, str]) -> tuple[list[str], list[str]]:
    train: list[str] = []
    test: list[str] = []
    seen_products: set[str] = set()

    for image_name in image_names:
        product_id = ground_truth.get(image_name)
        if not product_id:
            raise ValueError(f"Image '{image_name}' has no product_id in ground truth mapping.")

        product_id = str(product_id)
        if product_id not in seen_products:
            train.append(image_name)
            seen_products.add(product_id)
        else:
            test.append(image_name)

    return train, test


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split images into train/test where train keeps first image occurrence per product_id."
    )
    parser.add_argument(
        "--ground-truth",
        default="ground_truth.json",
        help="Path to filename->product_id mapping JSON.",
    )
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Directory containing image files.",
    )
    parser.add_argument(
        "--output",
        default="image_split_first_occurrence.json",
        help="Output JSON file path.",
    )
    args = parser.parse_args()

    ground_truth_path = Path(args.ground_truth)
    ground_truth_data = load_json(ground_truth_path)
    if not isinstance(ground_truth_data, dict):
        raise ValueError(f"Expected a JSON object in {ground_truth_path}")

    ground_truth: dict[str, str] = {str(filename): str(product_id) for filename, product_id in ground_truth_data.items()}
    image_names = collect_images(Path(args.images_dir))
    train, test = split_by_first_product_occurrence(image_names, ground_truth)

    output = {
        "summary": {
            "total_images": len(image_names),
            "train_images": len(train),
            "test_images": len(test),
            "unique_products_in_train": len(train),
        },
        "train": train,
        "test": test,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(
        f"Wrote split to {output_path}: "
        f"{len(train)} train images (first per product), {len(test)} test images (remaining occurrences)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
