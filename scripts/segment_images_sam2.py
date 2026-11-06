#!/usr/bin/env python3
"""Segment all images in images/ into regions using Meta SAM 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import pipeline


def collect_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )


def choose_device(device_arg: str) -> int:
    if device_arg == "auto":
        return 0 if torch.cuda.is_available() else -1
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return 0
    return -1


def mask_to_bool_array(mask: Any) -> np.ndarray:
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("L"))
    elif isinstance(mask, torch.Tensor):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)

    if array.ndim > 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {array.shape}")

    if array.dtype == bool:
        return array
    return array > 0


def extract_regions(raw_output: Any) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []

    if isinstance(raw_output, dict) and "masks" in raw_output:
        masks = raw_output.get("masks", [])
        scores = raw_output.get("scores", [])
        for index, mask in enumerate(masks):
            score = float(scores[index]) if index < len(scores) else None
            regions.append({"mask": mask_to_bool_array(mask), "score": score})
        return regions

    if isinstance(raw_output, list):
        for entry in raw_output:
            if isinstance(entry, dict) and "mask" in entry:
                score_value = entry.get("score")
                score = float(score_value) if score_value is not None else None
                regions.append({"mask": mask_to_bool_array(entry["mask"]), "score": score})
            elif isinstance(entry, dict) and "masks" in entry:
                regions.extend(extract_regions(entry))
        return regions

    if isinstance(raw_output, dict) and "mask" in raw_output:
        score_value = raw_output.get("score")
        score = float(score_value) if score_value is not None else None
        return [{"mask": mask_to_bool_array(raw_output["mask"]), "score": score}]

    raise RuntimeError(f"Unsupported SAM output format: {type(raw_output).__name__}")


def mask_bbox_and_area(mask: np.ndarray) -> tuple[list[int], int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0], 0
    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())
    area = int(mask.sum())
    return [x_min, y_min, x_max, y_max], area


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment images into regions using Meta SAM 2.")
    parser.add_argument("--images-dir", type=Path, default=Path("images"), help="Directory containing input images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sam2_segments"),
        help="Directory where masks and metadata are saved.",
    )
    parser.add_argument(
        "--model",
        default="facebook/sam2.1-hiera-large",
        help="Hugging Face model id for Meta SAM 2.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Number of images to process per batch.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device.")
    parser.add_argument("--min-region-area", type=int, default=200, help="Discard masks smaller than this many pixels.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N images (0 = all).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_paths = collect_images(args.images_dir)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError("No images found to segment.")

    device = choose_device(args.device)
    print(f"Using model: {args.model}")
    print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

    segmenter = pipeline(
        task="mask-generation",
        model=args.model,
        device=device,
    )

    output_dir = args.output_dir
    masks_root = output_dir / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "model": args.model,
        "device": "cuda" if device == 0 else "cpu",
        "images_processed": 0,
        "total_regions": 0,
        "images": [],
    }

    image_inputs = [str(path) for path in image_paths]
    effective_batch_size = args.batch_size
    if effective_batch_size != 1:
        print(
            "SAM2 mask-generation currently fails with batch_size > 1 in this Transformers build; "
            "falling back to batch_size=1 for stability."
        )
        effective_batch_size = 1

    outputs = segmenter(image_inputs, batch_size=effective_batch_size)
    if not isinstance(outputs, list):
        outputs = [outputs]

    for image_path, output in zip(image_paths, outputs):
        regions = extract_regions(output)
        image = Image.open(image_path)
        width, height = image.size
        image.close()

        region_dir = masks_root / image_path.stem
        region_dir.mkdir(parents=True, exist_ok=True)

        region_entries: list[dict[str, Any]] = []
        kept = 0
        for idx, region in enumerate(regions, start=1):
            mask_bool = region["mask"]
            bbox, area = mask_bbox_and_area(mask_bool)
            if area < args.min_region_area:
                continue

            mask_image = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
            mask_file = region_dir / f"mask_{idx:03d}.png"
            mask_image.save(mask_file)

            kept += 1
            summary["total_regions"] += 1
            region_entries.append(
                {
                    "mask_file": str(mask_file.as_posix()),
                    "score": round(region["score"], 6) if region["score"] is not None else None,
                    "bbox_xyxy": bbox,
                    "area_pixels": area,
                    "coverage_ratio": round(area / float(width * height), 6),
                }
            )

        summary["images"].append(
            {
                "image": str(image_path.as_posix()),
                "width": width,
                "height": height,
                "regions_count": kept,
                "regions": region_entries,
            }
        )
        summary["images_processed"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "segments_metadata.json"
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Processed {summary['images_processed']} images.")
    print(f"Saved {summary['total_regions']} regions to {masks_root}")
    print(f"Metadata written to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
