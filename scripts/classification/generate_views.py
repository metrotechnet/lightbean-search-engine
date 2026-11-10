#!/usr/bin/env python3
"""Generate texture views using Y-axis perspective + in-plane rotations.

Example:
    python scripts/classification/generate_views.py
    python scripts/classification/generate_views.py --input texture.jpg --output ./views
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_PLANE_ROT = [0, 180, 240, 271, 300]
DEFAULT_PERSPECTIVE_FACTORS = [0.85]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_PLANE_ROT_TEXT = ",".join(str(value) for value in DEFAULT_PLANE_ROT)
DEFAULT_PERSPECTIVE_FACTORS_TEXT = ",".join(str(value) for value in DEFAULT_PERSPECTIVE_FACTORS)


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


def largest_inbound_square_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    if h == 0 or w == 0:
        return None
    if not np.any(binary):
        return None

    dp = np.zeros((h, w), dtype=np.int32)
    best_side = 0
    best_row = 0
    best_col = 0

    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0:
                continue
            if x == 0 or y == 0:
                dp[y, x] = 1
            else:
                dp[y, x] = 1 + min(dp[y - 1, x], dp[y, x - 1], dp[y - 1, x - 1])

            if dp[y, x] > best_side:
                best_side = int(dp[y, x])
                best_row = y
                best_col = x

    if best_side <= 0:
        return None
    x2 = best_col + 1
    y2 = best_row + 1
    x1 = x2 - best_side
    y1 = y2 - best_side
    return (x1, y1, x2, y2)


def adjust_affine_to_positive_canvas(matrix: np.ndarray, width: int, height: int) -> tuple[np.ndarray, int, int]:
    corners = np.array(
        [[0.0, 0.0, 1.0], [width - 1.0, 0.0, 1.0], [width - 1.0, height - 1.0, 1.0], [0.0, height - 1.0, 1.0]],
        dtype=np.float64,
    )
    transformed = corners @ matrix.T
    min_x = float(np.min(transformed[:, 0]))
    min_y = float(np.min(transformed[:, 1]))
    max_x = float(np.max(transformed[:, 0]))
    max_y = float(np.max(transformed[:, 1]))
    out_w = max(1, int(math.ceil(max_x - min_x + 1.0)))
    out_h = max(1, int(math.ceil(max_y - min_y + 1.0)))
    adjusted = matrix.copy()
    adjusted[0, 2] -= min_x
    adjusted[1, 2] -= min_y
    return adjusted, out_w, out_h


def adjust_homography_to_positive_canvas(matrix: np.ndarray, width: int, height: int) -> tuple[np.ndarray, int, int]:
    corners = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    min_x = float(np.min(transformed[:, 0]))
    min_y = float(np.min(transformed[:, 1]))
    max_x = float(np.max(transformed[:, 0]))
    max_y = float(np.max(transformed[:, 1]))
    out_w = max(1, int(math.ceil(max_x - min_x + 1.0)))
    out_h = max(1, int(math.ceil(max_y - min_y + 1.0)))
    translation = np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]], dtype=np.float64)
    adjusted = translation @ matrix
    return adjusted, out_w, out_h


def apply_affine_with_mask(image: np.ndarray, mask: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    adjusted, out_w, out_h = adjust_affine_to_positive_canvas(matrix, w, h)
    warped_image = cv2.warpAffine(
        image,
        adjusted,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask,
        adjusted,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_image, warped_mask


def apply_perspective_with_mask(image: np.ndarray, mask: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    adjusted, out_w, out_h = adjust_homography_to_positive_canvas(matrix, w, h)
    warped_image = cv2.warpPerspective(
        image,
        adjusted,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpPerspective(
        mask,
        adjusted,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_image, warped_mask


def crop_square_from_valid_region(image: np.ndarray, valid_mask: np.ndarray, output_size: int) -> np.ndarray:
    ys, xs = np.where(valid_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)
    square_bbox = largest_inbound_square_bbox(valid_mask)
    if square_bbox is not None:
        x1, y1, x2, y2 = square_bbox
    else:
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)


def perspective_matrix(axis: str, perspective_factor: float, width: int, height: int) -> np.ndarray:
    src = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    factor = float(perspective_factor)
    if axis == "y":
        shift = float(abs(factor) * height * 0.45)
        if factor >= 0.0:
            dst = np.array(
                [[0.0, shift], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0 - shift]],
                dtype=np.float32,
            )
        else:
            dst = np.array(
                [[0.0, 0.0], [width - 1.0, shift], [width - 1.0, height - 1.0 - shift], [0.0, height - 1.0]],
                dtype=np.float32,
            )
    else:
        raise ValueError(f"Unsupported perspective axis '{axis}'.")
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


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
    parser = argparse.ArgumentParser(description="Generate Y-perspective + in-plane-rotation views of a texture image.")
    parser.add_argument("--input", type=Path, default=Path("texture/main_page"), help="Input texture image path or directory.")
    parser.add_argument("--output", type=Path, default=Path("texture/views"), help="Output directory for generated views.")
    parser.add_argument("--perspective-factor", default=DEFAULT_PERSPECTIVE_FACTORS_TEXT, help="Comma-separated perspective factors applied to Y perspective transform.")
    parser.add_argument("--plane-rot", default=DEFAULT_PLANE_ROT_TEXT, help="Comma-separated in-plane rotation angles in degrees (default: 0..360 step 30).")
    parser.add_argument("--size", type=int, default=224, help="Output image size (square).")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality for saved views.")
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


def generate_for_image(
    image_path: Path,
    output_dir: Path,
    perspective_factors: list[float],
    plane_rotations: list[float],
    size: int,
    quality: int,
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
    base_mask = np.full((image_bgr.shape[0], image_bgr.shape[1]), 255, dtype=np.uint8)
    for perspective_factor in perspective_factors:
        perspective_h = perspective_matrix("y", perspective_factor, image_bgr.shape[1], image_bgr.shape[0])
        perspective_img, perspective_mask = apply_perspective_with_mask(image_bgr, base_mask, perspective_h)
        center = (perspective_img.shape[1] * 0.5, perspective_img.shape[0] * 0.5)
        factor_label = safe_label_angle(perspective_factor).replace("-", "m")
        for rotation_deg in plane_rotations:
            rot_matrix = cv2.getRotationMatrix2D(center, float(rotation_deg), 1.0).astype(np.float64)
            rotated_img, rotated_mask = apply_affine_with_mask(perspective_img, perspective_mask, rot_matrix)
            projected = crop_square_from_valid_region(rotated_img, rotated_mask, output_size=size)
            angle_label = safe_label_angle(rotation_deg).zfill(3)
            file_name = f"perspy_pf{factor_label}_rot{angle_label}.jpg"
            out_path = output_dir / file_name
            cv2.imwrite(str(out_path), projected, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            contact_cells.append((f"persp y pf {factor_label} rot {angle_label}", projected))
            generated += 1

    contact_sheet = build_contact_sheet(
        cells=contact_cells,
        tile_size=size,
        columns=max(1, len(plane_rotations)),
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

    plane_rotations = parse_angle_list(args.plane_rot)
    perspective_factors = parse_angle_list(args.perspective_factor)
    if not perspective_factors:
        raise ValueError("--perspective-factor must contain at least one value.")
    for value in perspective_factors:
        if abs(float(value)) >= 1.0:
            raise ValueError(f"--perspective-factor value {value} is invalid; magnitude must be < 1.0.")

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    images = collect_input_images(input_path)
    if not images:
        raise RuntimeError(f"No supported images found in: {input_path}")

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
            perspective_factors=perspective_factors,
            plane_rotations=plane_rotations,
            size=args.size,
            quality=args.quality,
        )
        total_views += generated
        print(f"[{idx}/{len(images)}] Saved {generated} views for {image_path.name} -> {image_output_dir}")

    print(f"Completed. Processed {len(images)} image(s), generated {total_views} projected views in: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
