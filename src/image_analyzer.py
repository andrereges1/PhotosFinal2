"""Camada local de analise inteligente de imagem."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.analysis_models import ImageAnalysisReport
from src.config import (
    FRAME_PRIORITY_BALANCED,
    HIGH_RISK_CROP_PERCENT,
    ORIENTATION_HORIZONTAL,
    ORIENTATION_SQUARE,
    ORIENTATION_VERTICAL,
)
from src.face_detection import detect_faces_local
from src.local_vision import (
    Box,
    analyze_border_importance,
    calculate_center_importance,
    calculate_cover_crop_box,
    calculate_outer_border_importance,
    calculate_visual_complexity,
    classify_crop_amount,
    expand_box,
    is_box_inside_crop,
    is_box_near_edges,
    union_boxes,
)
from src.subject_crop import decide_subject_crop_strategy

logger = logging.getLogger(__name__)

YOLO_COMMON_OBJECTS = {
    "person",
    "dog",
    "cat",
    "book",
    "chair",
    "dining table",
    "couch",
    "potted plant",
    "vase",
    "tv",
    "backpack",
    "handbag",
    "cell phone",
    "teddy bear",
    "bottle",
    "cup",
    "laptop",
    "sports ball",
    "umbrella",
}

STRATEGY_SAFE_CROP = "safe_crop"
STRATEGY_SUBJECT_FOCUSED_CROP = "subject_focused_crop"
STRATEGY_CONTAIN = "contain_with_borders"
STRATEGY_CENTER_CROP = "center_crop"
STRATEGY_SMART_FACE_CROP = "smart_face_crop"
STRATEGY_CREATE_EXTRA_PAGE = "create_extra_page"
STRATEGY_MANUAL_REVIEW = "manual_review"

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"

_YOLO_MODEL: Any | None = None
_YOLO_MODEL_PATH: Path | None = None
_YOLO_MODEL_ATTEMPTED = False
_YOLO_MISSING_LOGGED = False


def _detect_orientation(width: int, height: int) -> str:
    if height > width * 1.10:
        return ORIENTATION_VERTICAL
    if width > height * 1.10:
        return ORIENTATION_HORIZONTAL
    return ORIENTATION_SQUARE


def _crop_requirement(
    image_size: tuple[int, int],
    target_aspect_ratio: float,
) -> tuple[str | None, float]:
    width, height = image_size
    image_aspect = width / height
    if abs(image_aspect - target_aspect_ratio) < 0.001:
        return None, 0.0

    if image_aspect > target_aspect_ratio:
        crop_width = int(round(height * target_aspect_ratio))
        removed = max(0, width - crop_width)
        return "width", removed / max(1, width) * 100

    crop_height = int(round(width / target_aspect_ratio))
    removed = max(0, height - crop_height)
    return "height", removed / max(1, height) * 100


def _resize_for_detection(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    image = image.convert("RGB")
    width, height = image.size
    largest = max(width, height)
    if largest <= max_side:
        return image.copy(), 1.0

    scale = max_side / largest
    resized_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(resized_size, Image.Resampling.LANCZOS), 1 / scale


def _find_local_yolo_model() -> Path | None:
    env_path = os.environ.get("FOTO_10X15_YOLO_MODEL")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            Path("models") / "yolo11n.pt",
            Path("models") / "yolov8n.pt",
            Path("models") / "best.pt",
            Path("yolo11n.pt"),
            Path("yolov8n.pt"),
            Path("best.pt"),
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _get_yolo_model() -> Any | None:
    global _YOLO_MODEL, _YOLO_MODEL_ATTEMPTED, _YOLO_MODEL_PATH, _YOLO_MISSING_LOGGED

    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    if _YOLO_MODEL_ATTEMPTED:
        return None

    _YOLO_MODEL_ATTEMPTED = True
    model_path = _find_local_yolo_model()
    if model_path is None:
        if not _YOLO_MISSING_LOGGED:
            logger.info("Modelo YOLO local nao configurado; analise segue sem YOLO.")
            _YOLO_MISSING_LOGGED = True
        return None

    try:
        from ultralytics import YOLO

        _YOLO_MODEL = YOLO(str(model_path))
        _YOLO_MODEL_PATH = model_path
        logger.info("Modelo YOLO local carregado de %s", model_path)
    except Exception:
        logger.info("Falha ao carregar modelo YOLO local; analise segue sem YOLO.")
        _YOLO_MODEL = None
    return _YOLO_MODEL


def detect_objects_yolo(image: Image.Image) -> list[dict[str, Any]]:
    model = _get_yolo_model()
    if model is None:
        return []

    try:
        rgb = image.convert("RGB")
        image_array = np.array(rgb)
        results = model.predict(source=image_array, verbose=False, conf=0.25)
    except Exception:
        logger.info("Falha na deteccao YOLO; analise segue sem YOLO.")
        return []

    detections: list[dict[str, Any]] = []
    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for raw_box in boxes:
            try:
                class_id = int(raw_box.cls[0])
                confidence = float(raw_box.conf[0])
                class_name = str(names.get(class_id, class_id))
                if class_name not in YOLO_COMMON_OBJECTS and confidence < 0.55:
                    continue
                x1, y1, x2, y2 = [float(value) for value in raw_box.xyxy[0].tolist()]
                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": confidence,
                        "x": max(0, int(round(x1))),
                        "y": max(0, int(round(y1))),
                        "w": max(1, int(round(x2 - x1))),
                        "h": max(1, int(round(y2 - y1))),
                        "source": "yolo",
                    }
                )
            except Exception:
                logger.info("Falha ao converter caixa YOLO; ignorando deteccao.")
                continue
    return detections


def detect_text_regions_tesseract(image: Image.Image) -> list[dict[str, Any]]:
    try:
        import pytesseract
        from pytesseract import Output
    except Exception:
        logger.info("pytesseract nao disponivel; analise segue sem OCR.")
        return []

    try:
        resized, scale_back = _resize_for_detection(image, max_side=1200)
        data = pytesseract.image_to_data(
            resized,
            output_type=Output.DICT,
            config="--psm 11",
        )
    except Exception:
        logger.info("Tesseract indisponivel ou nao configurado; analise segue sem OCR.")
        return []

    width, height = image.size
    boxes: list[dict[str, Any]] = []
    item_count = len(data.get("text", []))
    for index in range(item_count):
        text = str(data["text"][index]).strip()
        if not text:
            continue

        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < 45:
            continue

        x = int(round(float(data["left"][index]) * scale_back))
        y = int(round(float(data["top"][index]) * scale_back))
        box_width = int(round(float(data["width"][index]) * scale_back))
        box_height = int(round(float(data["height"][index]) * scale_back))
        if box_width <= 1 or box_height <= 1:
            continue

        box = {
            "text": text,
            "confidence": confidence,
            "x": max(0, x),
            "y": max(0, y),
            "w": max(1, box_width),
            "h": max(1, box_height),
        }
        box["near_edge"] = is_box_near_edges(box, width, height)
        boxes.append(box)
    return boxes


def _boxes_safe_for_crop(boxes: list[dict[str, Any]], crop_box: Box, image_width: int, image_height: int) -> bool:
    if not boxes:
        return True

    for box in boxes:
        expanded = expand_box(box, image_width, image_height, 0.12)
        if not is_box_inside_crop(expanded, crop_box):
            return False
    return True


def _focus_box(
    faces: list[dict[str, Any]],
    persons: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> Box | None:
    preferred_boxes = persons or faces
    group = union_boxes(preferred_boxes)
    if group is None:
        return None

    margin = 0.20
    if len(preferred_boxes) > 1:
        margin = 0.34
    group_width_ratio = group[2] / max(1, image_width)
    group_height_ratio = group[3] / max(1, image_height)
    if group_width_ratio > 0.60 or group_height_ratio > 0.60:
        margin = max(margin, 0.40)
    return expand_box(group, image_width, image_height, margin)


def _group_spread_score(group_box: Box | None, image_width: int, image_height: int) -> float:
    if group_box is None:
        return 0.0
    _, _, width, height = group_box
    return max(width / max(1, image_width), height / max(1, image_height))


def _append_unique(items: list[str], item: str) -> None:
    if item and item not in items:
        items.append(item)


def suggest_local_strategy(report: ImageAnalysisReport) -> str:
    reasons = list(report.reasons)
    warnings = list(report.warnings)
    strategy = STRATEGY_CONTAIN
    risk_level = RISK_MEDIUM

    has_faces_or_people = report.faces_detected > 0 or report.persons_detected > 0
    object_near_edges = any(
        is_box_near_edges(box, report.width, report.height)
        for box in report.object_boxes
        if box.get("class_name") != "person"
    )
    group_spread = max(
        _group_spread_score(report.face_group_box, report.width, report.height),
        _group_spread_score(report.person_group_box, report.width, report.height),
    )

    if report.required_crop_percent <= 0.1:
        strategy = STRATEGY_SAFE_CROP
        risk_level = RISK_LOW
        _append_unique(reasons, "A foto ja esta muito proxima do formato final.")
    elif report.required_crop_percent > HIGH_RISK_CROP_PERCENT:
        strategy = STRATEGY_CONTAIN
        risk_level = RISK_HIGH
        _append_unique(reasons, "Usei bordas porque o corte necessario seria grande.")
    elif report.faces_near_edges:
        strategy = STRATEGY_CONTAIN
        risk_level = RISK_HIGH
        _append_unique(reasons, "Usei bordas para evitar cortar pessoas.")
    elif report.persons_near_edges:
        strategy = STRATEGY_CONTAIN
        risk_level = RISK_HIGH
        if report.persons_detected > 1:
            _append_unique(reasons, "Detectei varias pessoas e preservei o grupo.")
        else:
            _append_unique(reasons, "Usei bordas para evitar cortar pessoas.")
    elif report.text_near_edges:
        strategy = STRATEGY_CONTAIN
        risk_level = RISK_HIGH
        _append_unique(reasons, "Usei bordas porque detectei texto perto da borda.")
    elif not report.faces_safe_for_crop or not report.persons_safe_for_crop:
        strategy = STRATEGY_CONTAIN
        risk_level = RISK_HIGH
        _append_unique(reasons, "Usei bordas para evitar cortar pessoas.")
    elif report.recommended_crop_mode == STRATEGY_SUBJECT_FOCUSED_CROP and report.can_tighten_frame:
        strategy = STRATEGY_SUBJECT_FOCUSED_CROP
        risk_level = RISK_LOW if report.background_waste_score >= 0.22 else RISK_MEDIUM
        _append_unique(
            reasons,
            report.subject_crop_reason or "Usei foco nas pessoas para cortar fundo irrelevante.",
        )
    elif (report.persons_detected > 1 or report.faces_detected > 1) and group_spread > 0.55:
        strategy = STRATEGY_CONTAIN
        risk_level = RISK_HIGH
        _append_unique(reasons, "Detectei varias pessoas e preservei o grupo.")
    elif report.edge_importance_max > 0.58:
        strategy = STRATEGY_CONTAIN
        risk_level = RISK_HIGH
        _append_unique(reasons, "Usei bordas para preservar detalhes importantes perto da borda.")
    elif object_near_edges and report.edge_importance_max > 0.35:
        strategy = STRATEGY_CONTAIN
        risk_level = RISK_MEDIUM
        _append_unique(reasons, "Usei bordas para preservar objetos perto da borda.")
    elif report.crop_amount_class in {"none", "tiny"}:
        strategy = STRATEGY_SAFE_CROP
        risk_level = RISK_LOW
        _append_unique(reasons, "Usei corte leve porque as bordas pareciam pouco importantes.")
    elif report.edge_importance_max <= 0.28 and report.crop_amount_class in {"small", "medium"}:
        strategy = STRATEGY_SMART_FACE_CROP if has_faces_or_people else STRATEGY_SAFE_CROP
        risk_level = RISK_LOW if report.crop_amount_class == "small" else RISK_MEDIUM
        _append_unique(reasons, "Usei corte leve porque as bordas pareciam pouco importantes.")
    elif not has_faces_or_people and not report.text_detected and report.crop_amount_class == "small":
        strategy = STRATEGY_SAFE_CROP
        risk_level = RISK_LOW
        _append_unique(reasons, "Usei corte leve porque nao detectei pessoas, texto ou detalhes importantes nas bordas.")
    elif report.edge_importance_max > 0.35:
        strategy = STRATEGY_CONTAIN
        risk_level = RISK_MEDIUM
        _append_unique(reasons, "Usei bordas para preservar detalhes importantes perto da borda.")
    else:
        strategy = STRATEGY_CONTAIN
        risk_level = RISK_MEDIUM
        _append_unique(reasons, "Usei bordas porque havia duvida sobre a seguranca do corte.")

    if report.visual_complexity_score > 0.72 and report.required_crop_percent > 8:
        risk_level = RISK_HIGH
        if strategy != STRATEGY_CONTAIN:
            strategy = STRATEGY_CONTAIN
        _append_unique(reasons, "Usei bordas porque a imagem tem muitos detalhes.")

    report.suggested_strategy = strategy
    report.risk_level = risk_level
    report.reasons = reasons
    report.warnings = warnings
    return strategy


def analyze_image_locally(
    image: Image.Image,
    image_name: str,
    target_format: str,
    target_size: tuple[int, int],
    frame_priority: str = FRAME_PRIORITY_BALANCED,
) -> ImageAnalysisReport:
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    target_aspect = target_size[0] / target_size[1]
    required_crop_axis, required_crop_percent = _crop_requirement(rgb_image.size, target_aspect)

    face_boxes = detect_faces_local(rgb_image)
    object_boxes = detect_objects_yolo(rgb_image)
    person_boxes = [box for box in object_boxes if box.get("class_name") == "person"]
    text_boxes = detect_text_regions_tesseract(rgb_image)

    focus = _focus_box(face_boxes, person_boxes, width, height)
    crop_box = calculate_cover_crop_box(rgb_image.size, target_aspect, focus)
    border_scores = analyze_border_importance(rgb_image, crop_box)

    face_group_box = union_boxes(face_boxes)
    person_group_box = union_boxes(person_boxes)
    faces_near_edges = any(is_box_near_edges(box, width, height) for box in face_boxes)
    persons_near_edges = any(is_box_near_edges(box, width, height) for box in person_boxes)
    text_near_edges = any(bool(box.get("near_edge")) for box in text_boxes)

    report = ImageAnalysisReport(
        image_name=image_name,
        width=width,
        height=height,
        orientation=_detect_orientation(width, height),
        aspect_ratio=width / max(1, height),
        target_format=target_format,
        target_aspect_ratio=target_aspect,
        required_crop_axis=required_crop_axis,
        required_crop_percent=required_crop_percent,
        crop_amount_class=classify_crop_amount(required_crop_percent, bool(face_boxes or person_boxes)),
        faces_detected=len(face_boxes),
        face_boxes=face_boxes,
        face_group_box=face_group_box,
        faces_near_edges=faces_near_edges,
        faces_safe_for_crop=_boxes_safe_for_crop(face_boxes, crop_box, width, height),
        persons_detected=len(person_boxes),
        person_boxes=person_boxes,
        person_group_box=person_group_box,
        persons_near_edges=persons_near_edges,
        persons_safe_for_crop=_boxes_safe_for_crop(person_boxes, crop_box, width, height),
        text_detected=bool(text_boxes),
        text_boxes=text_boxes,
        text_near_edges=text_near_edges,
        edge_importance_left=border_scores.get("left", 0.0),
        edge_importance_right=border_scores.get("right", 0.0),
        edge_importance_top=border_scores.get("top", 0.0),
        edge_importance_bottom=border_scores.get("bottom", 0.0),
        edge_importance_max=border_scores.get("max", 0.0),
        visual_complexity_score=calculate_visual_complexity(rgb_image),
        center_importance_score=calculate_center_importance(rgb_image),
        border_importance_score=calculate_outer_border_importance(rgb_image),
        suggested_strategy=STRATEGY_CONTAIN,
        risk_level=RISK_MEDIUM,
        reasons=[],
        warnings=[],
        objects_detected=len(object_boxes),
        object_boxes=object_boxes,
    )
    subject_decision = decide_subject_crop_strategy(
        rgb_image,
        report,
        face_boxes,
        person_boxes,
        object_boxes,
        text_boxes,
        target_size,
        frame_priority,
    )
    report.primary_subject_type = subject_decision.primary_subject_type
    report.primary_subject_box = subject_decision.primary_subject_box
    report.primary_subject_expanded_box = subject_decision.expanded_subject_box
    report.primary_subject_confidence = subject_decision.primary_subject_confidence
    report.subject_focus_score = subject_decision.subject_focus_score
    report.background_waste_score = subject_decision.background_waste_score
    report.empty_area_top_score = subject_decision.empty_area_scores.get("top", 0.0)
    report.empty_area_bottom_score = subject_decision.empty_area_scores.get("bottom", 0.0)
    report.empty_area_left_score = subject_decision.empty_area_scores.get("left", 0.0)
    report.empty_area_right_score = subject_decision.empty_area_scores.get("right", 0.0)
    report.can_tighten_frame = subject_decision.can_tighten_frame
    report.recommended_crop_mode = subject_decision.strategy
    report.subject_crop_reason = subject_decision.reason
    suggest_local_strategy(report)
    return report
