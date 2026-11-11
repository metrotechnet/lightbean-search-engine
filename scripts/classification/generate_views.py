#!/usr/bin/env python3
"""Generate texture views using scale + rotation transforms.

Example:
    python scripts/classification/generate_views.py
    python scripts/classification/generate_views.py --input texture.jpg --output ./views
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_SCALES = [0.5, 0.75, 1.0, 1.5]
DEFAULT_ROTATIONS = [0, 45, 90, 135]
DEFAULT_PERSPECTIVE_STRENGTH = 0.4
DEFAULT_PERSPECTIVE_DIRECTION = ["left", "right", "top"]
DEFAULT_PERSPECTIVE_DIRECTION_TEXT = ",".join(DEFAULT_PERSPECTIVE_DIRECTION)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_SCALES_TEXT = ",".join(str(value) for value in DEFAULT_SCALES)
DEFAULT_ROTATIONS_TEXT = ",".join(str(value) for value in DEFAULT_ROTATIONS)


def parse_angle_list(value: str) -> list[float]:
    parts = [token.strip() for token in str(value).split(",") if token.strip()]
    if not parts:
        raise ValueError("Angle list cannot be empty.")
    return [float(token) for token in parts]


def safe_label_angle(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value).replace(".", "p")


def safe_dir_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    cleaned = cleaned.strip("._")
    return cleaned or "image"


def simulate_zoom(image: np.ndarray, zoom: float) -> np.ndarray:
    h, w = image.shape[:2]

    if zoom >= 1:
        crop_w = max(1, round(w / zoom))
        crop_h = max(1, round(h / zoom))
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2
        result = image[y : y + crop_h, x : x + crop_w]
    else:
        target_w = round(w / zoom)
        target_h = round(h / zoom)
        repeats_x = int(np.ceil(target_w / w)) + 1
        repeats_y = int(np.ceil(target_h / h)) + 1
        tiled = np.tile(image, (repeats_y, repeats_x, 1))
        x = (tiled.shape[1] - target_w) // 2
        y = (tiled.shape[0] - target_h) // 2
        result = tiled[y : y + target_h, x : x + target_w]

    return cv2.resize(
        result,
        (w, h),
        interpolation=cv2.INTER_AREA if zoom < 1 else cv2.INTER_LINEAR,
    )


def largest_inbound_square_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    if h == 0 or w == 0:
        return None
    if not np.any(binary):
        return None
    if np.all(binary):
        return (0, 0, w, h)

    dist = cv2.distanceTransform(binary * 255, cv2.DIST_C, 3)
    _, max_dist, _, max_loc = cv2.minMaxLoc(dist)
    if max_dist <= 0.0:
        return None

    side = int(2 * int(max_dist) - 1)
    if side <= 0:
        side = 1
    cx, cy = int(max_loc[0]), int(max_loc[1])
    half = side // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)



def largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:

    """
    Find the largest axis-aligned rectangle fully inside
    the non-black image region.

    Returns:
        (x, y, width, height), cropped_image
    """
    square_bbox = largest_inbound_square_bbox(mask)
    #Reduce bbox until it fits within the mask
    if square_bbox is None:
        return None
    x1, y1, x2, y2 = square_bbox
    while x2 > x1 and y2 > y1 and not np.all(mask[y1:y2, x1:x2] > 0):
        x1 += 1
        y1 += 1
        x2 -= 1
        y2 -= 1
        if x2 <= x1 or y2 <= y1:
            return None
    return (x1, y1, x2, y2)


def perspective_view(image, strength=0.4, direction: str = "top"):
    """
    Apply a projective skew with a configurable vanishing direction.

    direction can be one of:
      - "top": vanishing toward the top edge
      - "right": vanishing toward the right edge
      - "bottom": vanishing toward the bottom edge
      - "left": vanishing toward the left edge
      - "none": no perspective
    """
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    valid_directions = {"top", "right", "bottom", "left", "none"}
    if direction not in valid_directions:
        raise ValueError(f"Unknown direction: {direction}")

    h, w = image.shape[:2]

    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1],
    ])

    shift = strength * max(w, h) * 0.45
    dst = src.copy()

    if direction == "top":
        dst[0, 0] = shift
        dst[1, 0] = w - 1 - shift
    elif direction == "right":
    # right edge narrows vertically
        dst[1, 1] = shift
        dst[2, 1] = h - 1 - shift
    elif direction == "bottom":
    # bottom edge narrows horizontally
        dst[2, 0] = w - 1 - shift
        dst[3, 0] = shift

    elif direction == "left":
    # left edge narrows vertically
        dst[0, 1] = shift
        dst[3, 1] = h - 1 - shift
    elif direction == "none":
        dst = src.copy()

    H = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(
        image,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    full_mask = np.ones((h, w), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(
        full_mask,
        H,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return warped, warped_mask



def transformed_square_view(
    image_bgr: np.ndarray,
    scale: float,
    rotation: float,
    output_size: int,
    perspective_direction: str = "left",
    perspective_strength: float = DEFAULT_PERSPECTIVE_STRENGTH,
) -> np.ndarray:

    # Appliquer un zoom simulé et une perspective vers le haut
    scaled = simulate_zoom(image_bgr, float(scale))
    scaled_height, scaled_width = scaled.shape[:2]
    scaled_mask = np.full((scaled.shape[0], scaled.shape[1]), 255, dtype=np.uint8)
    
    center = (scaled_width / 2.0, scaled_height / 2.0)
    rot_matrix = cv2.getRotationMatrix2D(center, float(rotation), 1.0)
    abs_cos = abs(rot_matrix[0, 0])
    abs_sin = abs(rot_matrix[0, 1])
    rotated_width = max(1, int(math.ceil((scaled_height * abs_sin) + (scaled_width * abs_cos))))
    rotated_height = max(1, int(math.ceil((scaled_height * abs_cos) + (scaled_width * abs_sin))))
    rot_matrix[0, 2] += (rotated_width / 2.0) - center[0]
    rot_matrix[1, 2] += (rotated_height / 2.0) - center[1]

    rotated = cv2.warpAffine(
        scaled,
        rot_matrix,
        (rotated_width, rotated_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
   
    rotated_mask = cv2.warpAffine(
        scaled_mask,
        rot_matrix,
        (rotated_width, rotated_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    
  
    square_bbox = largest_inbound_square_bbox(rotated_mask)
    if square_bbox is None:
        cropped = rotated
    else:
        x1, y1, x2, y2 = square_bbox
        cropped = rotated[y1:y2, x1:x2]
        if cropped.size == 0:
            cropped = rotated

    # After the perspective warp, keep only the valid in-bounds area and crop
    # its largest inscribed square before returning the final texture view.
    warped, warped_mask = perspective_view(
        cropped,
        strength=float(perspective_strength),
        direction=perspective_direction,
    )

    valid_bbox = largest_valid_rectangle(warped_mask)
    if valid_bbox is None:
        return warped
    # Draw the valid valid_bbox on the warped image for debugging purposes.
    # x1, y1, x2, y2 = valid_bbox
    # cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # return warped
    x1, y1, x2, y2 = valid_bbox
    cropped_valid = warped[y1:y2, x1:x2]
    resized_valid = cv2.resize(cropped_valid, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    return resized_valid


def build_contact_sheet(
    cells: list[tuple[str, np.ndarray]],
    tile_size: int,
    columns: int,
    title: str,
) -> np.ndarray:
    if not cells:
        raise ValueError("No cells were provided for contact sheet.")
    label_h = 28
    margin = 10
    cols = max(1, columns)
    rows = math.ceil(len(cells) / cols)
    title_h = 36

    width = margin * (cols + 1) + cols * tile_size
    height = title_h + margin + rows * (tile_size + label_h + margin)
    sheet = np.full((height, width, 3), 255, dtype=np.uint8)

    cv2.putText(sheet, title, (margin, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)

    for idx, (label, tile) in enumerate(cells):
        row = idx // cols
        col = idx % cols
        x = margin + col * (tile_size + margin)
        y = title_h + margin + row * (tile_size + label_h + margin)
        thumb = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        sheet[y : y + tile_size, x : x + tile_size] = thumb
        cv2.rectangle(sheet, (x, y), (x + tile_size, y + tile_size), (180, 180, 180), 1)
        cv2.putText(
            sheet,
            label,
            (x, y + tile_size + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scaled + rotated views of a texture image.")
    parser.add_argument("--input", type=Path, default=Path("texture/main_page"), help="Input texture image path or directory.")
    parser.add_argument("--output", type=Path, default=Path("texture/views"), help="Output directory for generated views.")
    parser.add_argument("--scales", default=DEFAULT_SCALES_TEXT, help="Comma-separated scale factors.")
    parser.add_argument("--rotations", default=DEFAULT_ROTATIONS_TEXT, help="Comma-separated rotation angles in degrees.")
    parser.add_argument("--size", type=int, default=224, help="Output image size (square).")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality for saved views.")
    parser.add_argument(
        "--perspective-directions",
        default=DEFAULT_PERSPECTIVE_DIRECTION_TEXT,
        help="Comma-separated perspective directions: left,right,top,...",
    )
    parser.add_argument(
        "--perspective-strength",
        type=float,
        default=DEFAULT_PERSPECTIVE_STRENGTH,
        help="Perspective strength used for each direction.",
    )
    return parser.parse_args()


def collect_input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported input file extension: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        images = [path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        return sorted(images)
    raise FileNotFoundError(f"Input path not found: {input_path}")


def compute_scaled_rotated_views(
    image_bgr: np.ndarray,
    scales: list[float],
    rotations: list[float],
    output_size: int,
    perspective_directions: list[str] | None = None,
    perspective_strength: float = DEFAULT_PERSPECTIVE_STRENGTH,
) -> list[dict[str, Any]]:
    if output_size <= 0:
        raise ValueError(f"output_size must be > 0, got {output_size}.")
    if not scales:
        raise ValueError("At least one scale is required.")
    if not rotations:
        raise ValueError("At least one rotation is required.")
    if any(scale <= 0 for scale in scales):
        raise ValueError(f"All scales must be > 0, got {scales}.")

    if perspective_directions is None:
        perspective_directions = list(DEFAULT_PERSPECTIVE_DIRECTION)

    output: list[dict[str, Any]] = []
    for scale in scales:
        for rotation in rotations:
            for direction in perspective_directions:
                transformed_bgr = transformed_square_view(
                    image_bgr=image_bgr,
                    scale=float(scale),
                    rotation=float(rotation),
                    output_size=output_size,
                    perspective_direction=direction,
                    perspective_strength=float(perspective_strength),
                )
                output.append(
                    {
                        "scale": float(scale),
                        "rotation": float(rotation),
                        "direction": direction,
                        "image_bgr": transformed_bgr,
                    }
                )
    return output


def generate_for_image(
    image_path: Path,
    output_dir: Path,
    scales: list[float],
    rotations: list[float],
    size: int,
    quality: int,
    perspective_directions: list[str] | None = None,
    perspective_strength: float = DEFAULT_PERSPECTIVE_STRENGTH,
) -> int:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    original_square = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    original_path = output_dir / f"original_{image_path.stem}.jpg"
    cv2.imwrite(str(original_path), original_square, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])

    contact_cells: list[tuple[str, np.ndarray]] = [("original", original_square)]
    generated = 0
    transformed_views = compute_scaled_rotated_views(
        image_bgr=image_bgr,
        scales=scales,
        rotations=rotations,
        output_size=size,
        perspective_directions=perspective_directions,
        perspective_strength=perspective_strength,
    )
    for view in transformed_views:
        scale_value = float(view["scale"])
        rotation_value = float(view["rotation"])
        direction_value = str(view["direction"])
        transformed_bgr = view["image_bgr"]
        scale_label = safe_label_angle(scale_value).replace("-", "m")
        rotation_label = safe_label_angle(rotation_value).zfill(3)
        file_name = f"scale{scale_label}_rot{rotation_label}_{direction_value}.jpg"
        out_path = output_dir / file_name
        cv2.imwrite(str(out_path), transformed_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        contact_cells.append((f"s {scale_label} r {rotation_label} {direction_value}", transformed_bgr))
        generated += 1

    contact_sheet = build_contact_sheet(
        cells=contact_cells,
        tile_size=size,
        columns=max(1, len(rotations)),
        title=f"Texture projections: {image_path.name}",
    )
    contact_sheet_path = output_dir / "contact_sheet.jpg"
    cv2.imwrite(str(contact_sheet_path), contact_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return generated


def main() -> int:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be > 0.")
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    scales = parse_angle_list(args.scales)
    rotations = parse_angle_list(args.rotations)
    if any(scale <= 0 for scale in scales):
        raise ValueError(f"All --scales values must be > 0, got {scales}.")

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    images = collect_input_images(input_path)
    if not images:
        raise RuntimeError(f"No supported images found in: {input_path}")

    perspective_directions = [direction.strip() for direction in str(args.perspective_directions).split(",") if direction.strip()]
    if not perspective_directions:
        perspective_directions = list(DEFAULT_PERSPECTIVE_DIRECTION)

    total_views = 0
    for idx, image_path in enumerate(images, start=1):
        if input_path.is_dir():
            rel = image_path.relative_to(input_path).as_posix()
            rel_dir = safe_dir_component(rel.rsplit(".", 1)[0].replace("/", "__"))
            image_output_dir = output_root / f"{idx:05d}_{rel_dir}"
        else:
            image_output_dir = output_root

        generated = generate_for_image(
            image_path=image_path,
            output_dir=image_output_dir,
            scales=scales,
            rotations=rotations,
            size=args.size,
            quality=args.quality,
            perspective_directions=perspective_directions,
            perspective_strength=args.perspective_strength,
        )
        total_views += generated
        print(f"[{idx}/{len(images)}] Saved {generated} views for {image_path.name} -> {image_output_dir}")

    print(f"Completed. Processed {len(images)} image(s), generated {total_views} projected views in: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
