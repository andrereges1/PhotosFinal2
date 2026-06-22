"""Heuristicas locais de visao computacional para corte seguro."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

Box = tuple[int, int, int, int]
CropBox = tuple[int, int, int, int]


def _to_rgb_array(region: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(region, Image.Image):
        return np.array(region.convert("RGB"))

    array = np.asarray(region)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    if array.ndim == 3 and array.shape[2] == 4:
        return array[:, :, :3]
    if array.ndim == 3 and array.shape[2] >= 3:
        return array[:, :, :3]
    return np.zeros((1, 1, 3), dtype=np.uint8)


def _resize_for_analysis(array: np.ndarray, max_width: int = 800) -> np.ndarray:
    height, width = array.shape[:2]
    if width <= max_width:
        return array

    scale = max_width / max(1, width)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(array, new_size, interpolation=cv2.INTER_AREA)


def _box_parts(box: dict[str, Any] | Box) -> Box:
    if isinstance(box, dict):
        return (
            int(box.get("x", 0)),
            int(box.get("y", 0)),
            int(box.get("w", 0)),
            int(box.get("h", 0)),
        )
    x, y, width, height = box
    return int(x), int(y), int(width), int(height)


def calculate_edge_density(region: Image.Image | np.ndarray) -> float:
    try:
        array = _resize_for_analysis(_to_rgb_array(region))
        if array.size == 0:
            return 0.0
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return float(np.count_nonzero(edges) / max(1, edges.size))
    except Exception:
        logger.exception("Falha ao calcular densidade de bordas")
        return 0.0


def calculate_color_variation(region: Image.Image | np.ndarray) -> float:
    try:
        array = _resize_for_analysis(_to_rgb_array(region))
        if array.size == 0:
            return 0.0
        channel_std = np.std(array.astype(np.float32), axis=(0, 1))
        return float(np.clip(np.mean(channel_std) / 128.0, 0.0, 1.0))
    except Exception:
        logger.exception("Falha ao calcular variacao de cor")
        return 0.0


def calculate_region_importance(region: Image.Image | np.ndarray) -> float:
    try:
        array = _resize_for_analysis(_to_rgb_array(region))
        if array.size == 0:
            return 0.0

        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges) / max(1, edges.size))
        color_variation = calculate_color_variation(array)
        brightness_variation = float(np.clip(np.std(gray.astype(np.float32)) / 128.0, 0.0, 1.0))

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_density = min(1.0, len(contours) / max(1.0, edges.size / 2500))

        importance = edge_density * 4.5 + color_variation * 0.32 + brightness_variation * 0.20 + contour_density * 0.20
        return float(np.clip(importance, 0.0, 1.0))
    except Exception:
        logger.exception("Falha ao calcular importancia da regiao")
        return 0.0


def get_crop_removed_regions(image: Image.Image, crop_box: CropBox) -> dict[str, Image.Image]:
    image = image.convert("RGB")
    crop_x, crop_y, crop_width, crop_height = crop_box
    image_width, image_height = image.size
    crop_right = crop_x + crop_width
    crop_bottom = crop_y + crop_height
    regions: dict[str, Image.Image] = {}

    if crop_x > 1:
        regions["left"] = image.crop((0, 0, crop_x, image_height))
    if crop_right < image_width - 1:
        regions["right"] = image.crop((crop_right, 0, image_width, image_height))
    if crop_y > 1:
        regions["top"] = image.crop((crop_x, 0, crop_right, crop_y))
    if crop_bottom < image_height - 1:
        regions["bottom"] = image.crop((crop_x, crop_bottom, crop_right, image_height))

    return {
        name: region
        for name, region in regions.items()
        if region.width > 1 and region.height > 1
    }


def analyze_border_importance(image: Image.Image, crop_box: CropBox) -> dict[str, float]:
    scores = {
        "left": 0.0,
        "right": 0.0,
        "top": 0.0,
        "bottom": 0.0,
    }
    for name, region in get_crop_removed_regions(image, crop_box).items():
        scores[name] = calculate_region_importance(region)
    scores["max"] = max(scores.values()) if scores else 0.0
    return scores


def calculate_visual_complexity(image: Image.Image | np.ndarray) -> float:
    edge_density = calculate_edge_density(image)
    color_variation = calculate_color_variation(image)
    return float(np.clip(edge_density * 3.5 + color_variation * 0.45, 0.0, 1.0))


def classify_crop_amount(crop_percent: float, has_people_or_faces: bool = False) -> str:
    if crop_percent <= 0.1:
        return "none"
    if crop_percent <= 2.0:
        return "tiny"
    if has_people_or_faces:
        if crop_percent <= 6.0:
            return "small"
        if crop_percent <= 8.0:
            return "medium"
        if crop_percent <= 20.0:
            return "large"
        return "too_large"
    if crop_percent <= 8.0:
        return "small"
    if crop_percent <= 12.0:
        return "medium"
    if crop_percent <= 20.0:
        return "large"
    return "too_large"


def is_box_near_edges(
    box: dict[str, Any] | Box,
    image_width: int,
    image_height: int,
    margin_percent: float = 0.08,
) -> bool:
    x, y, width, height = _box_parts(box)
    margin_x = image_width * margin_percent
    margin_y = image_height * margin_percent
    return (
        x <= margin_x
        or y <= margin_y
        or x + width >= image_width - margin_x
        or y + height >= image_height - margin_y
    )


def union_boxes(boxes: Iterable[dict[str, Any] | Box]) -> Box | None:
    boxes_list = [_box_parts(box) for box in boxes]
    boxes_list = [box for box in boxes_list if box[2] > 0 and box[3] > 0]
    if not boxes_list:
        return None

    left = min(x for x, _, _, _ in boxes_list)
    top = min(y for _, y, _, _ in boxes_list)
    right = max(x + width for x, _, width, _ in boxes_list)
    bottom = max(y + height for _, y, _, height in boxes_list)
    return (left, top, right - left, bottom - top)


def expand_box(
    box: dict[str, Any] | Box,
    image_width: int,
    image_height: int,
    margin_percent: float,
) -> Box:
    x, y, width, height = _box_parts(box)
    margin_x = width * margin_percent
    margin_y = height * margin_percent
    left = max(0, int(round(x - margin_x)))
    top = max(0, int(round(y - margin_y)))
    right = min(image_width, int(round(x + width + margin_x)))
    bottom = min(image_height, int(round(y + height + margin_y)))
    return (left, top, max(1, right - left), max(1, bottom - top))


def calculate_cover_crop_box(
    image_size: tuple[int, int],
    target_aspect_ratio: float,
    focus_box: Box | None = None,
) -> CropBox:
    image_width, image_height = image_size
    image_aspect = image_width / image_height

    if image_aspect > target_aspect_ratio:
        crop_height = image_height
        crop_width = int(round(crop_height * target_aspect_ratio))
    else:
        crop_width = image_width
        crop_height = int(round(crop_width / target_aspect_ratio))

    crop_width = max(1, min(crop_width, image_width))
    crop_height = max(1, min(crop_height, image_height))

    if focus_box is None:
        crop_x = int(round((image_width - crop_width) / 2))
        crop_y = int(round((image_height - crop_height) / 2))
    else:
        focus_x, focus_y, focus_width, focus_height = focus_box
        focus_center_x = focus_x + focus_width / 2
        focus_center_y = focus_y + focus_height / 2
        crop_x = int(round(focus_center_x - crop_width / 2))
        crop_y = int(round(focus_center_y - crop_height / 2))

        if focus_width <= crop_width:
            if focus_x < crop_x:
                crop_x = focus_x
            if focus_x + focus_width > crop_x + crop_width:
                crop_x = focus_x + focus_width - crop_width

        if focus_height <= crop_height:
            if focus_y < crop_y:
                crop_y = focus_y
            if focus_y + focus_height > crop_y + crop_height:
                crop_y = focus_y + focus_height - crop_height

    crop_x = max(0, min(crop_x, image_width - crop_width))
    crop_y = max(0, min(crop_y, image_height - crop_height))
    return (crop_x, crop_y, crop_width, crop_height)


def is_box_inside_crop(
    box: dict[str, Any] | Box,
    crop_box: CropBox,
    safety_margin_px: int = 0,
) -> bool:
    x, y, width, height = _box_parts(box)
    crop_x, crop_y, crop_width, crop_height = crop_box
    return (
        x >= crop_x + safety_margin_px
        and y >= crop_y + safety_margin_px
        and x + width <= crop_x + crop_width - safety_margin_px
        and y + height <= crop_y + crop_height - safety_margin_px
    )


def calculate_center_importance(image: Image.Image) -> float:
    image = image.convert("RGB")
    width, height = image.size
    left = int(width * 0.25)
    top = int(height * 0.25)
    right = int(width * 0.75)
    bottom = int(height * 0.75)
    return calculate_region_importance(image.crop((left, top, right, bottom)))


def calculate_outer_border_importance(image: Image.Image, margin_percent: float = 0.10) -> float:
    image = image.convert("RGB")
    width, height = image.size
    margin_x = max(1, int(round(width * margin_percent)))
    margin_y = max(1, int(round(height * margin_percent)))
    regions = [
        image.crop((0, 0, margin_x, height)),
        image.crop((width - margin_x, 0, width, height)),
        image.crop((0, 0, width, margin_y)),
        image.crop((0, height - margin_y, width, height)),
    ]
    return max(calculate_region_importance(region) for region in regions)
