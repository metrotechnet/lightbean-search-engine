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

DEFAULT_PROMPT_MAX_COMPONENTS = 6
DEFAULT_PROMPT_MIN_COMPONENT_AREA = 64
DEFAULT_PROMPT_IOU_DEDUPE = 0.98


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_sam2_assets(model_id: str) -> tuple[Path, Path]:
    root = project_root()
    model_key = model_id.lower()
    if "hiera-large" in model_key:
        config_name = "sam2.1_hiera_l.yaml"
        checkpoint_name = "sam2.1_hiera_large.pt"
    elif "hiera-base-plus" in model_key:
        config_name = "sam2.1_hiera_b+.yaml"
        checkpoint_name = "sam2.1_hiera_base_plus.pt"
    elif "hiera-small" in model_key:
        config_name = "sam2.1_hiera_s.yaml"
        checkpoint_name = "sam2.1_hiera_small.pt"
    elif "hiera-tiny" in model_key:
        config_name = "sam2.1_hiera_t.yaml"
        checkpoint_name = "sam2.1_hiera_tiny.pt"
    else:
        config_name = "sam2.1_hiera_l.yaml"
        checkpoint_name = "sam2.1_hiera_large.pt"

    config_path = root / "configs" / "sam2.1" / config_name
    checkpoint_path = root / "checkpoints" / checkpoint_name
    return config_path, checkpoint_path


def collect_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )


def resolve_images(images_dir: Path, single_image: Path | None, limit: int) -> list[Path]:
    if single_image is not None:
        if not single_image.exists() or not single_image.is_file():
            raise FileNotFoundError(f"Image file not found: {single_image}")
        if single_image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError(f"Unsupported image extension: {single_image.suffix}")
        return [single_image]

    image_paths = collect_images(images_dir)
    if limit > 0:
        image_paths = image_paths[:limit]
    return image_paths


def choose_device(device_arg: str) -> int:
    if device_arg == "auto":
        return 0 if torch.cuda.is_available() else -1
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print("[sam2] CUDA requested but not available; falling back to CPU.")
            return -1
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


def resize_mask(mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_size:
        return mask

    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    resized = mask_image.resize(target_size, resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def create_colored_overlay(image_rgb: np.ndarray, masks: list[np.ndarray], alpha: float = 0.45) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    palette = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0],
            [128, 0, 255],
        ],
        dtype=np.float32,
    )
    for idx, mask in enumerate(masks):
        color = palette[idx % len(palette)]
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_otsu_threshold(gray_image: np.ndarray) -> int:
    hist = np.bincount(gray_image.ravel(), minlength=256).astype(np.float64)
    total = float(gray_image.size)
    cumulative_weight = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256, dtype=np.float64))
    global_mean = cumulative_mean[-1] / total if total > 0 else 0.0

    numerator = (global_mean * cumulative_weight - cumulative_mean) ** 2
    denominator = cumulative_weight * (total - cumulative_weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(denominator > 0, numerator / denominator, 0.0)
    return int(np.argmax(sigma))


def extract_component_prompts_and_mask(
    binary_mask: np.ndarray,
    max_components: int,
    min_area: int,
) -> tuple[list[dict[str, int]], np.ndarray]:
    height, width = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    labels = np.zeros((height, width), dtype=np.int32)
    components: list[dict[str, int]] = []
    next_label = 1

    for y in range(height):
        for x in range(width):
            if not binary_mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True

            area = 0
            sum_x = 0
            sum_y = 0
            x_min = x
            y_min = y
            x_max = x
            y_max = y

            while stack:
                cy, cx = stack.pop()
                labels[cy, cx] = next_label
                area += 1
                sum_x += cx
                sum_y += cy
                x_min = min(x_min, cx)
                y_min = min(y_min, cy)
                x_max = max(x_max, cx)
                y_max = max(y_max, cy)

                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not binary_mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

            if area < min_area:
                continue
            components.append(
                {
                    "label": next_label,
                    "area": area,
                    "cx": int(round(sum_x / area)),
                    "cy": int(round(sum_y / area)),
                    "x1": x_min,
                    "y1": y_min,
                    "x2": x_max + 1,
                    "y2": y_max + 1,
                }
            )
            next_label += 1

    components.sort(key=lambda item: item["area"], reverse=True)
    selected = components[:max_components]
    selected_labels = [component["label"] for component in selected]
    filtered_mask = np.isin(labels, selected_labels)

    prompts: list[dict[str, int]] = []
    for component in selected:
        prompts.append(
            {
                "area": component["area"],
                "cx": component["cx"],
                "cy": component["cy"],
                "x1": component["x1"],
                "y1": component["y1"],
                "x2": component["x2"],
                "y2": component["y2"],
            }
        )
    return prompts, filtered_mask


def remove_small_connected_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    height, width = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    filtered = np.zeros_like(binary_mask, dtype=bool)

    for y in range(height):
        for x in range(width):
            if not binary_mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component_pixels: list[tuple[int, int]] = []

            while stack:
                cy, cx = stack.pop()
                component_pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not binary_mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

            if len(component_pixels) < min_area:
                continue
            for py, px in component_pixels:
                filtered[py, px] = True

    return filtered


def split_connected_components(binary_mask: np.ndarray) -> list[np.ndarray]:
    height, width = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    components: list[tuple[int, np.ndarray]] = []

    for y in range(height):
        for x in range(width):
            if not binary_mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component_mask = np.zeros_like(binary_mask, dtype=bool)
            area = 0

            while stack:
                cy, cx = stack.pop()
                component_mask[cy, cx] = True
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not binary_mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))

            components.append((area, component_mask))

    components.sort(key=lambda item: item[0], reverse=True)
    return [component_mask for _, component_mask in components]


def build_bright_dark_prompts(
    gray_image: np.ndarray,
    coverage_small: np.ndarray,
    max_components: int,
    min_area: int,
) -> tuple[list[dict[str, int]], list[dict[str, int]], dict[str, int], np.ndarray, np.ndarray]:
    dark_cutoff = 128
    bright_cutoff = 128
    
    remaining_small = ~coverage_small
    dark_mask = (gray_image <= dark_cutoff) & remaining_small
    bright_mask = (gray_image >= bright_cutoff) & remaining_small

    dark_prompts, dark_connected_mask = extract_component_prompts_and_mask(
        dark_mask,
        max_components=max_components,
        min_area=min_area,
    )
    bright_prompts, bright_connected_mask = extract_component_prompts_and_mask(
        bright_mask,
        max_components=max_components,
        min_area=min_area,
    )
    bright_connected_mask &= ~dark_connected_mask
    dark_connected_mask = remove_small_connected_components(dark_connected_mask, min_area=min_area)
    bright_connected_mask = remove_small_connected_components(bright_connected_mask, min_area=min_area)
    stats = {
        "dark_cutoff": dark_cutoff,
        "bright_cutoff": bright_cutoff,
        "remaining_pixels": int(remaining_small.sum()),
        "dark_remaining_pixels": int(dark_connected_mask.sum()),
        "bright_remaining_pixels": int(bright_connected_mask.sum()),
    }
    return dark_prompts, bright_prompts, stats, dark_connected_mask, bright_connected_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment images into regions using Meta SAM 2.")
    default_model_id = "facebook/sam2.1-hiera-large"
    parser.add_argument("--images-dir", type=Path, default=Path("images"), help="Directory containing input images.")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional single image to process. When set, only this image is segmented and overlay is displayed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sam2_segments"),
        help="Directory where masks and metadata are saved.",
    )
    parser.add_argument(
        "--model",
        default=default_model_id,
        help="Hugging Face model id for Meta SAM 2.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Number of images to process per batch.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device.")
    parser.add_argument("--min-region-area", type=int, default=200, help="Discard masks smaller than this many pixels.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N images (0 = all).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if DEFAULT_PROMPT_MAX_COMPONENTS < 1:
        raise ValueError(f"DEFAULT_PROMPT_MAX_COMPONENTS must be >= 1, got {DEFAULT_PROMPT_MAX_COMPONENTS}")
    if DEFAULT_PROMPT_MIN_COMPONENT_AREA < 1:
        raise ValueError(
            f"DEFAULT_PROMPT_MIN_COMPONENT_AREA must be >= 1, got {DEFAULT_PROMPT_MIN_COMPONENT_AREA}"
        )
    if DEFAULT_PROMPT_IOU_DEDUPE < 0 or DEFAULT_PROMPT_IOU_DEDUPE > 1:
        raise ValueError(f"DEFAULT_PROMPT_IOU_DEDUPE must be between 0 and 1, got {DEFAULT_PROMPT_IOU_DEDUPE}")

    image_paths = resolve_images(args.images_dir, args.image, args.limit)
    if not image_paths:
        raise RuntimeError("No images found to segment.")

    device = choose_device(args.device)
    device_name = "cuda" if device == 0 else "cpu"
    print(f"Using model: {args.model}")
    print(f"Using device: {device_name}")

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

    image_jobs: list[dict[str, Any]] = []
    resized_inputs: list[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
            reduced_size = (max(1, width // 4), max(1, height // 4))
            print(
                f"[sam2] Processing image: {image_path.name} | "
                f"original={width}x{height} | resized={reduced_size[0]}x{reduced_size[1]}"
            )
            resized_image = image.resize(reduced_size, resample=Image.Resampling.BILINEAR)
            image_jobs.append(
                {
                    "image_path": image_path,
                    "reduced_size": reduced_size,
                    "resized_image": resized_image,
                }
            )
            resized_inputs.append(resized_image)

    try:
        batched_outputs = segmenter(resized_inputs, batch_size=args.batch_size)
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        if args.batch_size <= 1:
            raise RuntimeError("SAM2 mask-generation failed with batch_size=1.") from exc
        print(
            "[sam2] Batch inference failed; retrying with batch_size=1 for compatibility. "
            f"Original error: {exc.__class__.__name__}: {exc}"
        )
        batched_outputs = segmenter(resized_inputs, batch_size=1)
    if not isinstance(batched_outputs, list):
        batched_outputs = [batched_outputs]
    if len(batched_outputs) != len(image_jobs):
        raise RuntimeError(
            f"Unexpected SAM output count: expected {len(image_jobs)} outputs, got {len(batched_outputs)}."
        )

    outputs: list[Any] = []
    for job, base_output in zip(image_jobs, batched_outputs):
        image_path = job["image_path"]
        reduced_size = job["reduced_size"]
        resized_image = job["resized_image"]

        base_regions = extract_regions(base_output)
        regions: list[dict[str, Any]] = [
            {"mask": region["mask"], "score": region["score"], "source": "automatic"} for region in base_regions
        ]

        coverage_small = np.zeros((reduced_size[1], reduced_size[0]), dtype=bool)
        for base_region in base_regions:
            coverage_small |= resize_mask(base_region["mask"], (reduced_size[0], reduced_size[1]))

        _, _, prompt_stats, dark_mask_small, bright_mask_small = build_bright_dark_prompts(
            gray_image=np.asarray(resized_image.convert("L")),
            coverage_small=coverage_small,
            max_components=DEFAULT_PROMPT_MAX_COMPONENTS,
            min_area=DEFAULT_PROMPT_MIN_COMPONENT_AREA,
        )

        dark_components = split_connected_components(dark_mask_small)
        bright_components = split_connected_components(bright_mask_small)

        for component_index, component_mask in enumerate(dark_components, start=1):
            regions.append(
                {
                    "mask": component_mask,
                    "score": None,
                    "source": f"remaining_dark_{component_index:03d}",
                }
            )
        for component_index, component_mask in enumerate(bright_components, start=1):
            regions.append(
                {
                    "mask": component_mask,
                    "score": None,
                    "source": f"remaining_bright_{component_index:03d}",
                }
            )

        prompt_stats["dark_component_regions"] = len(dark_components)
        prompt_stats["bright_component_regions"] = len(bright_components)

        print(
            f"[sam2] Remaining-area dark/bright for {image_path.name}: "
            f"remaining={prompt_stats['remaining_pixels']} dark={prompt_stats['dark_remaining_pixels']} "
            f"bright={prompt_stats['bright_remaining_pixels']} dark_regions={prompt_stats['dark_component_regions']} "
            f"bright_regions={prompt_stats['bright_component_regions']}"
        )

        outputs.append(
            {
                "regions": regions,
                "prompt_stats": prompt_stats,
                "dark_mask_small": dark_mask_small,
                "bright_mask_small": bright_mask_small,
            }
        )

    for image_path, output_data in zip(image_paths, outputs):
        print(f"[sam2] Segmenting masks for: {image_path.name}")
        regions = output_data["regions"]
        prompt_stats = output_data["prompt_stats"]
        dark_mask_small = output_data["dark_mask_small"]
        bright_mask_small = output_data["bright_mask_small"]
        with Image.open(image_path) as image:
            base_image = image.convert("RGB")
            width, height = base_image.size

        region_dir = masks_root / image_path.stem
        region_dir.mkdir(parents=True, exist_ok=True)

        dark_mask_bool = resize_mask(dark_mask_small, (width, height)) if dark_mask_small is not None else np.zeros((height, width), dtype=bool)
        bright_mask_bool = resize_mask(bright_mask_small, (width, height)) if bright_mask_small is not None else np.zeros((height, width), dtype=bool)
        dark_mask_path = region_dir / "dark_mask.png"
        bright_mask_path = region_dir / "bright_mask.png"
        Image.fromarray((dark_mask_bool.astype(np.uint8) * 255), mode="L").save(dark_mask_path)
        Image.fromarray((bright_mask_bool.astype(np.uint8) * 255), mode="L").save(bright_mask_path)
        print(f"[sam2] Saved dark mask: {dark_mask_path}")
        print(f"[sam2] Saved bright mask: {bright_mask_path}")

        region_entries: list[dict[str, Any]] = []
        kept_masks: list[np.ndarray] = []
        kept = 0
        for idx, region in enumerate(regions, start=1):
            mask_bool = resize_mask(region["mask"], (width, height))
            bbox, area = mask_bbox_and_area(mask_bool)
            if area < args.min_region_area:
                continue

            mask_image = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
            mask_file = region_dir / f"mask_{idx:03d}.png"
            mask_image.save(mask_file)

            kept += 1
            kept_masks.append(mask_bool)
            summary["total_regions"] += 1
            region_entries.append(
                {
                    "mask_file": str(mask_file.as_posix()),
                    "score": round(region["score"], 6) if region["score"] is not None else None,
                    "source": region.get("source", "automatic"),
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
                "prompt_stats": prompt_stats,
                "dark_mask_file": str(dark_mask_path.as_posix()),
                "bright_mask_file": str(bright_mask_path.as_posix()),
                "regions": region_entries,
            }
        )
        summary["images_processed"] += 1

        if args.image is not None and kept_masks:
            overlay_array = create_colored_overlay(np.asarray(base_image), kept_masks)
            overlay_image = Image.fromarray(overlay_array, mode="RGB")
            overlay_path = output_dir / f"{image_path.stem}_overlay.png"
            overlay_image.save(overlay_path)
            print(f"[sam2] Saved overlay preview: {overlay_path}")
            overlay_image.show(title=f"SAM2 Overlay - {image_path.name}")
            Image.fromarray((dark_mask_bool.astype(np.uint8) * 255), mode="L").show(title=f"SAM2 Dark Mask - {image_path.name}")
            Image.fromarray((bright_mask_bool.astype(np.uint8) * 255), mode="L").show(title=f"SAM2 Bright Mask - {image_path.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "segments_metadata.json"
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Processed {summary['images_processed']} images.")
    print(f"Saved {summary['total_regions']} regions to {masks_root}")
    print(f"Metadata written to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
