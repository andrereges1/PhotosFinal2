"""Recorte focado no assunto principal, especialmente pessoas e grupos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image

from src.analysis_models import ImageAnalysisReport
from src.config import (
    FRAME_PRIORITY_KEEP_SCENE,
    FRAME_PRIORITY_PEOPLE,
    KEEP_SCENE_MARGIN_BOTTOM,
    KEEP_SCENE_MARGIN_LEFT,
    KEEP_SCENE_MARGIN_RIGHT,
    KEEP_SCENE_MARGIN_TOP,
    MAX_EMPTY_BACKGROUND_SCORE_FOR_CROP,
    MAX_SUBJECT_CROP_PERCENT_BALANCED,
    MAX_SUBJECT_CROP_PERCENT_KEEP_SCENE,
    MAX_SUBJECT_CROP_PERCENT_PEOPLE,
    PEOPLE_FOCUS_MARGIN_BOTTOM,
    PEOPLE_FOCUS_MARGIN_LEFT,
    PEOPLE_FOCUS_MARGIN_RIGHT,
    PEOPLE_FOCUS_MARGIN_TOP,
    PRIMARY_SUBJECT_GROUP_PEOPLE,
    PRIMARY_SUBJECT_MIXED_SCENE,
    PRIMARY_SUBJECT_OBJECT,
    PRIMARY_SUBJECT_PEOPLE_WITH_PET,
    PRIMARY_SUBJECT_SINGLE_PERSON,
    PRIMARY_SUBJECT_UNKNOWN,
    STRATEGY_CONTAIN_BORDERS,
    STRATEGY_SAFE_CROP,
    STRATEGY_SUBJECT_FOCUSED_CROP,
    DEFAULT_SUBJECT_MARGIN_BOTTOM,
    DEFAULT_SUBJECT_MARGIN_LEFT,
    DEFAULT_SUBJECT_MARGIN_RIGHT,
    DEFAULT_SUBJECT_MARGIN_TOP,
)
from src.local_vision import Box, calculate_region_importance, union_boxes

CropBox = tuple[int, int, int, int]
PET_CLASSES = {"dog", "cat"}
PERSON_CLASS = "person"


@dataclass(slots=True)
class SubjectDetectionResult:
    primary_subject_type: str
    primary_subject_box: Box | None
    expanded_subject_box: Box | None
    included_boxes: list[dict[str, Any]] = field(default_factory=list)
    excluded_boxes: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class SubjectCropValidation:
    valid: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SubjectCropDecision:
    strategy: str
    crop_box: CropBox | None
    reason: str
    warnings: list[str] = field(default_factory=list)
    primary_subject_type: str = PRIMARY_SUBJECT_UNKNOWN
    primary_subject_box: Box | None = None
    expanded_subject_box: Box | None = None
    primary_subject_confidence: float = 0.0
    subject_focus_score: float = 0.0
    background_waste_score: float = 0.0
    empty_area_scores: dict[str, float] = field(default_factory=dict)
    can_tighten_frame: bool = False
    validation: SubjectCropValidation | None = None


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


def _box_to_dict(box: dict[str, Any] | Box, class_name: str = "unknown", weight: float = 0.0) -> dict[str, Any]:
    x, y, width, height = _box_parts(box)
    if isinstance(box, dict):
        result = dict(box)
    else:
        result = {"x": x, "y": y, "w": width, "h": height}
    result.setdefault("class_name", class_name)
    result.setdefault("weight", weight)
    return result


def _xyxy(box: dict[str, Any] | Box) -> tuple[int, int, int, int]:
    x, y, width, height = _box_parts(box)
    return x, y, x + width, y + height


def _intersects_or_close(primary_box: Box, secondary_box: Box, max_distance_px: float) -> bool:
    left_a, top_a, right_a, bottom_a = _xyxy(primary_box)
    left_b, top_b, right_b, bottom_b = _xyxy(secondary_box)
    if left_a <= right_b and right_a >= left_b and top_a <= bottom_b and bottom_a >= top_b:
        return True
    dx = max(left_a - right_b, left_b - right_a, 0)
    dy = max(top_a - bottom_b, top_b - bottom_a, 0)
    return (dx * dx + dy * dy) ** 0.5 <= max_distance_px


def is_secondary_subject_near_primary(
    primary_box: Box,
    secondary_box: Box,
    image_size: tuple[int, int] | None = None,
    max_distance_ratio: float = 0.18,
) -> bool:
    if image_size:
        reference = max(image_size)
    else:
        reference = max(primary_box[2], primary_box[3], secondary_box[2], secondary_box[3])
    return _intersects_or_close(primary_box, secondary_box, reference * max_distance_ratio)


def include_nearby_secondary_subjects(
    primary_box: Box,
    secondary_boxes: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for box in secondary_boxes:
        if is_secondary_subject_near_primary(primary_box, _box_parts(box), image_size):
            included.append(box)
        else:
            excluded.append(box)
    return included, excluded


def _face_likely_inside_person(face_box: dict[str, Any], person_boxes: list[dict[str, Any]]) -> bool:
    face_left, face_top, face_right, face_bottom = _xyxy(face_box)
    for person_box in person_boxes:
        left, top, right, bottom = _xyxy(person_box)
        if face_left >= left and face_top >= top and face_right <= right and face_bottom <= bottom:
            return True
    return False


def build_subject_union_box(
    faces: list[dict[str, Any]],
    persons: list[dict[str, Any]],
    pets: list[dict[str, Any]],
    text_regions: list[dict[str, Any]],
    image_size: tuple[int, int],
    frame_priority: str,
) -> tuple[Box | None, list[dict[str, Any]], list[dict[str, Any]], str, float, str]:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    subject_type = PRIMARY_SUBJECT_UNKNOWN
    confidence = 0.0
    reason = "Nao detectei assunto principal."

    if persons:
        included.extend(_box_to_dict(box, PERSON_CLASS, 0.95) for box in persons)
        person_union = union_boxes(included)
        if person_union is not None:
            nearby_pets, far_pets = include_nearby_secondary_subjects(person_union, pets, image_size)
            included.extend(_box_to_dict(box, str(box.get("class_name", "pet")), 0.85) for box in nearby_pets)
            excluded.extend(far_pets)
            for face in faces:
                if not _face_likely_inside_person(face, persons):
                    included.append(_box_to_dict(face, "face", 1.0))
            subject_type = PRIMARY_SUBJECT_PEOPLE_WITH_PET if nearby_pets else (
                PRIMARY_SUBJECT_GROUP_PEOPLE if len(persons) > 1 else PRIMARY_SUBJECT_SINGLE_PERSON
            )
            confidence = 0.90 if len(persons) > 1 else 0.84
            reason = "Usei as pessoas detectadas como assunto principal."
    elif faces:
        included.extend(_box_to_dict(box, "face", 1.0) for box in faces)
        subject_type = PRIMARY_SUBJECT_GROUP_PEOPLE if len(faces) > 1 else PRIMARY_SUBJECT_SINGLE_PERSON
        confidence = 0.76 if len(faces) > 1 else 0.68
        reason = "Usei os rostos detectados como referencia do assunto."
    elif text_regions:
        included.extend(_box_to_dict(box, "text", 0.80) for box in text_regions)
        subject_type = PRIMARY_SUBJECT_OBJECT
        confidence = 0.46
        reason = "Detectei texto; mantive comportamento conservador."

    subject_box = union_boxes(included)
    return subject_box, included, excluded, subject_type, confidence, reason


def _margins_for_priority(frame_priority: str) -> tuple[float, float, float, float]:
    if frame_priority == FRAME_PRIORITY_PEOPLE:
        return (
            PEOPLE_FOCUS_MARGIN_LEFT,
            PEOPLE_FOCUS_MARGIN_RIGHT,
            PEOPLE_FOCUS_MARGIN_TOP,
            PEOPLE_FOCUS_MARGIN_BOTTOM,
        )
    if frame_priority == FRAME_PRIORITY_KEEP_SCENE:
        return (
            KEEP_SCENE_MARGIN_LEFT,
            KEEP_SCENE_MARGIN_RIGHT,
            KEEP_SCENE_MARGIN_TOP,
            KEEP_SCENE_MARGIN_BOTTOM,
        )
    return (
        DEFAULT_SUBJECT_MARGIN_LEFT,
        DEFAULT_SUBJECT_MARGIN_RIGHT,
        DEFAULT_SUBJECT_MARGIN_TOP,
        DEFAULT_SUBJECT_MARGIN_BOTTOM,
    )


def expand_subject_box(
    subject_box: Box,
    image_size: tuple[int, int],
    frame_priority: str,
) -> Box:
    image_width, image_height = image_size
    x, y, width, height = subject_box
    margin_left, margin_right, margin_top, margin_bottom = _margins_for_priority(frame_priority)
    left = max(0, int(round(x - width * margin_left)))
    top = max(0, int(round(y - height * margin_top)))
    right = min(image_width, int(round(x + width + width * margin_right)))
    bottom = min(image_height, int(round(y + height + height * margin_bottom)))
    return left, top, max(1, right - left), max(1, bottom - top)


def detect_primary_subject(
    image: Image.Image,
    faces: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    text_regions: list[dict[str, Any]],
    frame_priority: str,
) -> SubjectDetectionResult:
    image_size = image.size
    persons = [box for box in objects if box.get("class_name") == PERSON_CLASS]
    pets = [box for box in objects if str(box.get("class_name", "")).lower() in PET_CLASSES]
    subject_box, included, excluded, subject_type, confidence, reason = build_subject_union_box(
        faces,
        persons,
        pets,
        text_regions,
        image_size,
        frame_priority,
    )
    expanded = expand_subject_box(subject_box, image_size, frame_priority) if subject_box else None
    return SubjectDetectionResult(
        primary_subject_type=subject_type,
        primary_subject_box=subject_box,
        expanded_subject_box=expanded,
        included_boxes=included,
        excluded_boxes=excluded,
        confidence=confidence,
        reason=reason,
    )


def _region_score(image: Image.Image, box: tuple[int, int, int, int]) -> float:
    left, top, right, bottom = box
    if right - left <= 2 or bottom - top <= 2:
        return 0.0
    region = image.crop((left, top, right, bottom))
    area_ratio = ((right - left) * (bottom - top)) / max(1, image.width * image.height)
    importance = calculate_region_importance(region)
    return float(np.clip(area_ratio * (1.0 - importance) * 3.0, 0.0, 1.0))


def calculate_empty_area_scores(image: Image.Image, subject_box: Box | None) -> dict[str, float]:
    if subject_box is None:
        return {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}
    x, y, width, height = subject_box
    right = x + width
    bottom = y + height
    image_width, image_height = image.size
    return {
        "top": _region_score(image, (0, 0, image_width, y)),
        "bottom": _region_score(image, (0, bottom, image_width, image_height)),
        "left": _region_score(image, (0, y, x, bottom)),
        "right": _region_score(image, (right, y, image_width, bottom)),
    }


def calculate_background_waste_score(image: Image.Image, subject_box: Box | None) -> float:
    scores = calculate_empty_area_scores(image, subject_box)
    if not scores:
        return 0.0
    return float(np.clip(max(scores.values()) * 0.65 + sum(scores.values()) / len(scores) * 0.35, 0.0, 1.0))


def _max_subject_crop_percent(frame_priority: str) -> float:
    if frame_priority == FRAME_PRIORITY_PEOPLE:
        return MAX_SUBJECT_CROP_PERCENT_PEOPLE
    if frame_priority == FRAME_PRIORITY_KEEP_SCENE:
        return MAX_SUBJECT_CROP_PERCENT_KEEP_SCENE
    return MAX_SUBJECT_CROP_PERCENT_BALANCED


def calculate_subject_focused_crop_box(
    image_size: tuple[int, int],
    target_aspect_ratio: float,
    subject_box: Box,
    expanded_subject_box: Box,
    frame_priority: str,
) -> CropBox | None:
    image_width, image_height = image_size
    expanded_x, expanded_y, expanded_width, expanded_height = expanded_subject_box
    if expanded_width <= 0 or expanded_height <= 0:
        return None

    expanded_aspect = expanded_width / expanded_height
    if expanded_aspect > target_aspect_ratio:
        crop_width = expanded_width
        crop_height = int(round(crop_width / target_aspect_ratio))
    else:
        crop_height = expanded_height
        crop_width = int(round(crop_height * target_aspect_ratio))

    crop_width = max(expanded_width, crop_width)
    crop_height = max(expanded_height, crop_height)

    if crop_width > image_width or crop_height > image_height:
        image_aspect = image_width / image_height
        if image_aspect > target_aspect_ratio:
            crop_height = image_height
            crop_width = int(round(crop_height * target_aspect_ratio))
        else:
            crop_width = image_width
            crop_height = int(round(crop_width / target_aspect_ratio))
        if crop_width < expanded_width or crop_height < expanded_height:
            return None

    subject_x, subject_y, subject_width, subject_height = subject_box
    center_x = subject_x + subject_width / 2
    center_y = subject_y + subject_height / 2
    crop_x = int(round(center_x - crop_width / 2))
    crop_y = int(round(center_y - crop_height / 2))

    if expanded_x < crop_x:
        crop_x = expanded_x
    if expanded_x + expanded_width > crop_x + crop_width:
        crop_x = expanded_x + expanded_width - crop_width
    if expanded_y < crop_y:
        crop_y = expanded_y
    if expanded_y + expanded_height > crop_y + crop_height:
        crop_y = expanded_y + expanded_height - crop_height

    crop_x = max(0, min(crop_x, image_width - crop_width))
    crop_y = max(0, min(crop_y, image_height - crop_height))
    candidate = (int(crop_x), int(crop_y), int(crop_width), int(crop_height))
    if not _box_inside_crop(expanded_subject_box, candidate):
        return None
    return candidate


def _box_inside_crop(box: dict[str, Any] | Box, crop_box: CropBox, margin_px: int = 0) -> bool:
    left, top, right, bottom = _xyxy(box)
    crop_x, crop_y, crop_width, crop_height = crop_box
    return (
        left >= crop_x + margin_px
        and top >= crop_y + margin_px
        and right <= crop_x + crop_width - margin_px
        and bottom <= crop_y + crop_height - margin_px
    )


def _largest_crop_size_for_aspect(image_size: tuple[int, int], target_aspect_ratio: float) -> tuple[int, int]:
    image_width, image_height = image_size
    image_aspect = image_width / max(1, image_height)
    if image_aspect > target_aspect_ratio:
        crop_height = image_height
        crop_width = int(round(crop_height * target_aspect_ratio))
    else:
        crop_width = image_width
        crop_height = int(round(crop_width / target_aspect_ratio))
    return max(1, min(crop_width, image_width)), max(1, min(crop_height, image_height))


def _clamp_aspect_crop(
    center_x: float,
    center_y: float,
    crop_width: int,
    crop_height: int,
    image_size: tuple[int, int],
) -> CropBox:
    image_width, image_height = image_size
    crop_width = max(1, min(crop_width, image_width))
    crop_height = max(1, min(crop_height, image_height))
    crop_x = int(round(center_x - crop_width / 2))
    crop_y = int(round(center_y - crop_height / 2))
    crop_x = max(0, min(crop_x, image_width - crop_width))
    crop_y = max(0, min(crop_y, image_height - crop_height))
    return crop_x, crop_y, crop_width, crop_height


def _adjust_crop_box_to_aspect(
    crop_box: CropBox,
    image_size: tuple[int, int],
    target_aspect_ratio: float,
) -> CropBox:
    crop_x, crop_y, crop_width, crop_height = crop_box
    center_x = crop_x + crop_width / 2
    center_y = crop_y + crop_height / 2
    current_aspect = crop_width / max(1, crop_height)

    if abs(current_aspect - target_aspect_ratio) <= 0.002:
        return _clamp_aspect_crop(center_x, center_y, crop_width, crop_height, image_size)

    if current_aspect > target_aspect_ratio:
        adjusted_width = crop_width
        adjusted_height = int(round(adjusted_width / target_aspect_ratio))
    else:
        adjusted_height = crop_height
        adjusted_width = int(round(adjusted_height * target_aspect_ratio))

    image_width, image_height = image_size
    if adjusted_width > image_width or adjusted_height > image_height:
        adjusted_width, adjusted_height = _largest_crop_size_for_aspect(image_size, target_aspect_ratio)

    return _clamp_aspect_crop(center_x, center_y, adjusted_width, adjusted_height, image_size)


def _important_text_near_subject(text_region: dict[str, Any], subject_box: Box | None, image_size: tuple[int, int]) -> bool:
    if subject_box is None:
        return True
    return is_secondary_subject_near_primary(subject_box, _box_parts(text_region), image_size, max_distance_ratio=0.10)


def validate_subject_crop(
    crop_box: CropBox,
    image_size: tuple[int, int],
    faces: list[dict[str, Any]],
    persons: list[dict[str, Any]],
    pets: list[dict[str, Any]],
    text_regions: list[dict[str, Any]],
    subject_box: Box | None,
    expanded_subject_box: Box | None,
    frame_priority: str,
) -> SubjectCropValidation:
    reasons: list[str] = []
    warnings: list[str] = []
    image_width, image_height = image_size
    crop_x, crop_y, crop_width, crop_height = crop_box

    if crop_x < 0 or crop_y < 0 or crop_width <= 0 or crop_height <= 0:
        return SubjectCropValidation(False, ["Crop invalido."], warnings)
    if crop_x + crop_width > image_width or crop_y + crop_height > image_height:
        return SubjectCropValidation(False, ["Crop saiu dos limites da imagem."], warnings)
    if expanded_subject_box and not _box_inside_crop(expanded_subject_box, crop_box):
        return SubjectCropValidation(False, ["O assunto principal nao coube no recorte."], warnings)

    subject_margin_px = max(4, int(round(min(crop_width, crop_height) * 0.015)))
    for face in faces:
        face_margin = max(3, int(round(min(_box_parts(face)[2], _box_parts(face)[3]) * 0.12)))
        if not _box_inside_crop(face, crop_box, face_margin):
            return SubjectCropValidation(False, ["Um rosto ficaria perto demais do corte."], warnings)

    for person in persons:
        if not _box_inside_crop(person, crop_box, subject_margin_px):
            return SubjectCropValidation(False, ["Uma pessoa ficaria cortada ou apertada."], warnings)

    for pet in pets:
        if subject_box and is_secondary_subject_near_primary(subject_box, _box_parts(pet), image_size):
            if not _box_inside_crop(pet, crop_box, subject_margin_px):
                return SubjectCropValidation(False, ["Um pet junto ao grupo ficaria cortado."], warnings)

    for text_region in text_regions:
        if _important_text_near_subject(text_region, subject_box, image_size) and not _box_inside_crop(text_region, crop_box):
            return SubjectCropValidation(False, ["Texto perto do assunto ficaria fora do recorte."], warnings)

    removed_percent = (1.0 - (crop_width * crop_height) / max(1, image_width * image_height)) * 100
    if removed_percent > _max_subject_crop_percent(frame_priority):
        return SubjectCropValidation(False, ["O recorte removeria cenario demais para esta prioridade."], warnings)

    if expanded_subject_box:
        _, _, expanded_width, expanded_height = expanded_subject_box
        if expanded_width > crop_width * 0.97 or expanded_height > crop_height * 0.97:
            return SubjectCropValidation(False, ["O recorte ficou apertado demais no assunto."], warnings)

    reasons.append("Recorte validado: assunto principal preservado.")
    return SubjectCropValidation(True, reasons, warnings)


def _grow_crop_box(
    crop_box: CropBox,
    image_size: tuple[int, int],
    grow_percent: float,
    target_aspect_ratio: float | None = None,
) -> CropBox:
    image_width, image_height = image_size
    crop_x, crop_y, crop_width, crop_height = crop_box
    center_x = crop_x + crop_width / 2
    center_y = crop_y + crop_height / 2

    if target_aspect_ratio is not None:
        desired_width = crop_width * (1.0 + grow_percent)
        desired_height = crop_height * (1.0 + grow_percent)
        desired_width = max(desired_width, desired_height * target_aspect_ratio)
        desired_height = desired_width / target_aspect_ratio
        if desired_width > image_width or desired_height > image_height:
            desired_width, desired_height = _largest_crop_size_for_aspect(image_size, target_aspect_ratio)
        return _clamp_aspect_crop(
            center_x,
            center_y,
            int(round(desired_width)),
            int(round(desired_height)),
            image_size,
        )

    grow_x = crop_width * grow_percent
    grow_y = crop_height * grow_percent
    left = max(0, int(round(crop_x - grow_x / 2)))
    top = max(0, int(round(crop_y - grow_y / 2)))
    right = min(image_width, int(round(crop_x + crop_width + grow_x / 2)))
    bottom = min(image_height, int(round(crop_y + crop_height + grow_y / 2)))
    return left, top, max(1, right - left), max(1, bottom - top)


def relax_crop_until_safe(
    crop_box: CropBox,
    image_size: tuple[int, int],
    validation_function: Callable[[CropBox], SubjectCropValidation],
    max_attempts: int = 5,
    target_aspect_ratio: float | None = None,
) -> CropBox | None:
    candidate = (
        _adjust_crop_box_to_aspect(crop_box, image_size, target_aspect_ratio)
        if target_aspect_ratio is not None
        else crop_box
    )
    for attempt in range(max_attempts + 1):
        validation = validation_function(candidate)
        if validation.valid:
            return candidate
        candidate = _grow_crop_box(candidate, image_size, 0.05 * (attempt + 1), target_aspect_ratio)
    return None


def _subject_focus_score(detection: SubjectDetectionResult, image_size: tuple[int, int]) -> float:
    if detection.primary_subject_box is None:
        return 0.0
    _, _, width, height = detection.primary_subject_box
    subject_area_ratio = (width * height) / max(1, image_size[0] * image_size[1])
    area_component = float(np.clip(1.0 - abs(subject_area_ratio - 0.42), 0.0, 1.0))
    return float(np.clip(detection.confidence * 0.75 + area_component * 0.25, 0.0, 1.0))


def decide_subject_crop_strategy(
    image: Image.Image,
    image_report: ImageAnalysisReport | None,
    faces: list[dict[str, Any]],
    persons: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    text_regions: list[dict[str, Any]],
    target_size: tuple[int, int],
    frame_priority: str,
) -> SubjectCropDecision:
    rgb_image = image.convert("RGB")
    target_aspect = target_size[0] / target_size[1]
    detection = detect_primary_subject(rgb_image, faces, objects, text_regions, frame_priority)
    empty_scores = calculate_empty_area_scores(rgb_image, detection.primary_subject_box)
    background_waste = calculate_background_waste_score(rgb_image, detection.primary_subject_box)
    focus_score = _subject_focus_score(detection, rgb_image.size)

    persons_or_faces = bool(persons or faces)
    if not persons_or_faces:
        return SubjectCropDecision(
            strategy=STRATEGY_SAFE_CROP,
            crop_box=None,
            reason="Nao detectei pessoas como assunto principal; mantive corte seguro.",
            primary_subject_type=detection.primary_subject_type,
            primary_subject_box=detection.primary_subject_box,
            expanded_subject_box=detection.expanded_subject_box,
            primary_subject_confidence=detection.confidence,
            subject_focus_score=focus_score,
            background_waste_score=background_waste,
            empty_area_scores=empty_scores,
            can_tighten_frame=False,
        )

    if detection.primary_subject_box is None or detection.expanded_subject_box is None:
        return SubjectCropDecision(
            strategy=STRATEGY_SAFE_CROP,
            crop_box=None,
            reason="Nao consegui criar uma caixa segura para o assunto.",
            primary_subject_type=detection.primary_subject_type,
            background_waste_score=background_waste,
            empty_area_scores=empty_scores,
        )

    if frame_priority == FRAME_PRIORITY_KEEP_SCENE:
        waste_threshold = 0.30
    elif frame_priority == FRAME_PRIORITY_PEOPLE:
        waste_threshold = 0.08
    else:
        waste_threshold = 0.12
    can_tighten = focus_score >= 0.55 and background_waste >= min(MAX_EMPTY_BACKGROUND_SCORE_FOR_CROP, waste_threshold)

    if not can_tighten:
        return SubjectCropDecision(
            strategy=STRATEGY_SAFE_CROP,
            crop_box=None,
            reason="A foto ja parece bem enquadrada ou sem fundo sobrando suficiente.",
            primary_subject_type=detection.primary_subject_type,
            primary_subject_box=detection.primary_subject_box,
            expanded_subject_box=detection.expanded_subject_box,
            primary_subject_confidence=detection.confidence,
            subject_focus_score=focus_score,
            background_waste_score=background_waste,
            empty_area_scores=empty_scores,
            can_tighten_frame=False,
        )

    crop_box = calculate_subject_focused_crop_box(
        rgb_image.size,
        target_aspect,
        detection.primary_subject_box,
        detection.expanded_subject_box,
        frame_priority,
    )
    if crop_box is None:
        return SubjectCropDecision(
            strategy=STRATEGY_CONTAIN_BORDERS,
            crop_box=None,
            reason="Nao consegui recortar o assunto no formato final com seguranca.",
            primary_subject_type=detection.primary_subject_type,
            primary_subject_box=detection.primary_subject_box,
            expanded_subject_box=detection.expanded_subject_box,
            primary_subject_confidence=detection.confidence,
            subject_focus_score=focus_score,
            background_waste_score=background_waste,
            empty_area_scores=empty_scores,
            can_tighten_frame=True,
        )

    pets = [box for box in objects if str(box.get("class_name", "")).lower() in PET_CLASSES]

    def validate(candidate: CropBox) -> SubjectCropValidation:
        return validate_subject_crop(
            candidate,
            rgb_image.size,
            faces,
            persons,
            pets,
            text_regions,
            detection.primary_subject_box,
            detection.expanded_subject_box,
            frame_priority,
        )

    relaxed = relax_crop_until_safe(crop_box, rgb_image.size, validate, target_aspect_ratio=target_aspect)
    if relaxed is None:
        validation = validate(crop_box)
        return SubjectCropDecision(
            strategy=STRATEGY_CONTAIN_BORDERS,
            crop_box=None,
            reason="Usei bordas porque o recorte por assunto ficou arriscado.",
            warnings=validation.reasons,
            primary_subject_type=detection.primary_subject_type,
            primary_subject_box=detection.primary_subject_box,
            expanded_subject_box=detection.expanded_subject_box,
            primary_subject_confidence=detection.confidence,
            subject_focus_score=focus_score,
            background_waste_score=background_waste,
            empty_area_scores=empty_scores,
            can_tighten_frame=True,
            validation=validation,
        )

    validation = validate(relaxed)
    return SubjectCropDecision(
        strategy=STRATEGY_SUBJECT_FOCUSED_CROP,
        crop_box=relaxed,
        reason="Usei foco nas pessoas e cortei apenas partes irrelevantes do fundo.",
        warnings=validation.warnings,
        primary_subject_type=detection.primary_subject_type,
        primary_subject_box=detection.primary_subject_box,
        expanded_subject_box=detection.expanded_subject_box,
        primary_subject_confidence=detection.confidence,
        subject_focus_score=focus_score,
        background_waste_score=background_waste,
        empty_area_scores=empty_scores,
        can_tighten_frame=True,
        validation=validation,
    )


def apply_subject_focused_crop(
    image: Image.Image,
    crop_box: CropBox,
    target_size: tuple[int, int],
) -> Image.Image:
    target_aspect = target_size[0] / max(1, target_size[1])
    crop_box = _adjust_crop_box_to_aspect(crop_box, image.size, target_aspect)
    crop_x, crop_y, crop_width, crop_height = crop_box
    crop_x = max(0, min(crop_x, image.width - 1))
    crop_y = max(0, min(crop_y, image.height - 1))
    crop_width = max(1, min(crop_width, image.width - crop_x))
    crop_height = max(1, min(crop_height, image.height - crop_y))
    cropped = image.convert("RGB").crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
    return cropped.resize(target_size, Image.Resampling.LANCZOS).convert("RGB")
