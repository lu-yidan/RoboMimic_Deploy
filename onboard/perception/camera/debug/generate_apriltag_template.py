"""Generate printable AprilTag templates on an A4 page.

Examples:
    python onboard/perception/camera/debug/generate_apriltag_template.py
    python onboard/perception/camera/debug/generate_apriltag_template.py --tag-id 5 --tag-size-mm 80
    python onboard/perception/camera/debug/generate_apriltag_template.py --family tag25h9 --tag-id 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


FAMILY_TO_DICT = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def _mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def _build_marker(family: str, tag_id: int, side_px: int) -> np.ndarray:
    if family not in FAMILY_TO_DICT:
        raise ValueError(
            f"Unsupported family '{family}'. Choose one of: {sorted(FAMILY_TO_DICT)}"
        )
    marker_dict = cv2.aruco.getPredefinedDictionary(FAMILY_TO_DICT[family])
    marker = np.zeros((side_px, side_px), dtype=np.uint8)
    # OpenCV generates the square marker image directly; the caller is
    # responsible for printing it at 100% scale so the physical edge matches
    # --tag-size used by apriltag_detector.py.
    cv2.aruco.generateImageMarker(marker_dict, int(tag_id), side_px, marker, 1)
    return marker


def _draw_text_center(page: np.ndarray, text: str, y: int, scale: float, thickness: int = 2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (page.shape[1] - w) // 2)
    cv2.putText(page, text, (x, y + h), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return y + h + baseline


def main():
    parser = argparse.ArgumentParser(description="Generate a printable A4 AprilTag template.")
    parser.add_argument("--family", default="tag36h11", help="AprilTag family, e.g. tag36h11.")
    parser.add_argument("--tag-id", type=int, default=0, help="AprilTag id to print.")
    parser.add_argument("--tag-size-mm", type=float, default=80.0,
                        help="Tag edge length in millimetres. This is the black square size.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI.")
    parser.add_argument("--margin-mm", type=float, default=15.0, help="Minimum page margin in millimetres.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    args = parser.parse_args()

    family = str(args.family).strip().lower()
    dpi = int(args.dpi)
    page_w = _mm_to_px(210.0, dpi)
    page_h = _mm_to_px(297.0, dpi)
    margin_px = _mm_to_px(args.margin_mm, dpi)
    tag_side_px = _mm_to_px(args.tag_size_mm, dpi)

    if tag_side_px <= 0:
        raise ValueError("--tag-size-mm must be positive.")
    if tag_side_px + 2 * margin_px > min(page_w, page_h):
        raise ValueError("Tag is too large for the requested A4 layout and margin.")

    page = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    marker = _build_marker(family, args.tag_id, tag_side_px)
    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    top_y = _mm_to_px(18.0, dpi)
    next_y = _draw_text_center(page, "AprilTag Printable Template", top_y, scale=0.9, thickness=2)
    next_y += _mm_to_px(3.0, dpi)
    next_y = _draw_text_center(
        page,
        f"family={family}   id={args.tag_id}   tag_size={args.tag_size_mm:.1f} mm",
        next_y,
        scale=0.65,
        thickness=2,
    )
    next_y += _mm_to_px(6.0, dpi)

    x0 = (page_w - tag_side_px) // 2
    y0 = max(next_y, (page_h - tag_side_px) // 2 - _mm_to_px(8.0, dpi))
    y0 = min(y0, page_h - margin_px - tag_side_px - _mm_to_px(35.0, dpi))
    page[y0:y0 + tag_side_px, x0:x0 + tag_side_px] = marker_bgr

    guide_y = y0 + tag_side_px + _mm_to_px(10.0, dpi)
    guide_y = _draw_text_center(
        page,
        "Measure the black square edge only. Use that value for --tag-size.",
        guide_y,
        scale=0.6,
        thickness=2,
    )
    guide_y += _mm_to_px(2.0, dpi)
    _draw_text_center(
        page,
        "Print at 100% scale on A4. Do not fit-to-page.",
        guide_y,
        scale=0.6,
        thickness=2,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_name = f"apriltag_{family}_id{args.tag_id}_{int(round(args.tag_size_mm))}mm_a4.png"
        output_path = Path(__file__).resolve().parent / output_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), page)
    if not ok:
        raise RuntimeError(f"Failed to write image to {output_path}")

    print(f"[INFO] Wrote printable template to: {output_path}")


if __name__ == "__main__":
    main()
