"""Processamento local das fotos, sem distorcer imagens."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import logging
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.ai_decision import get_ai_decision_for_report
from src.analysis_models import AIDecision, ImageAnalysisReport
from src.config import (
    AI_ALLOWED_DECISIONS,
    DEFAULT_MAX_SAFE_CROP_PERCENT,
    DPI,
    FORMAT_10X15_VERTICAL,
    FORMAT_15X10_HORIZONTAL,
    FORMAT_AUTO,
    FRAME_PRIORITY_KEEP_SCENE,
    HIGH_RISK_CROP_PERCENT,
    MAX_EDGE_IMPORTANCE_FOR_SAFE_CROP,
    ORIENTATION_HORIZONTAL,
    ORIENTATION_SQUARE,
    ORIENTATION_VERTICAL,
    RESIZE_CONTAIN,
    RESIZE_COVER,
    RESIZE_SAFE_CROP,
    RESIZE_MODE_LABELS,
    RESIZE_SMART,
    SAFE_CROP_DECISION_CONTAIN,
    SAFE_CROP_DECISION_CROP,
    STRICT_PEOPLE_SAFE_CROP_PERCENT,
    TARGET_10X15_PX,
    TARGET_15X10_PX,
    TARGET_LABELS,
    WHITE,
)
from src.face_detection import (
    FaceBox,
    calculate_smart_crop_box,
    detect_faces,
    is_crop_safe_for_faces,
)
from src.image_analyzer import (
    STRATEGY_CENTER_CROP,
    STRATEGY_CONTAIN,
    STRATEGY_CREATE_EXTRA_PAGE,
    STRATEGY_MANUAL_REVIEW,
    STRATEGY_SAFE_CROP,
    STRATEGY_SMART_FACE_CROP,
    STRATEGY_SUBJECT_FOCUSED_CROP,
    analyze_image_locally,
)
from src.subject_crop import apply_subject_focused_crop, decide_subject_crop_strategy
from src.utils import ProcessingOptions, ProcessedImage, merge_warnings, sanitize_filename

logger = logging.getLogger(__name__)
RESAMPLE = Image.Resampling.LANCZOS
CropBox = tuple[int, int, int, int]


@dataclass(slots=True)
class CropRequirements:
    crop_axis: str
    crop_pixels_total: int
    crop_pixels_each_side: float
    crop_percent_axis: float
    crop_percent_total: float
    crop_percent_width: float
    crop_percent_height: float
    original_aspect: float
    target_aspect: float


@dataclass(slots=True)
class EdgeImportance:
    edge_importance_score: float
    edge_density: float
    color_variation: float
    contour_density: float
    risk_level: str


@dataclass(slots=True)
class SafeCropDecision:
    decision: str
    reason: str
    crop_box: CropBox | None
    crop_percent_total: float
    crop_percent_width: float
    crop_percent_height: float
    edge_importance_score: float
    safe_crop_score: float
    warning: str | None = None


def open_image_safely(uploaded_file: BinaryIO) -> Image.Image:
    """Abre JPG, JPEG ou PNG e já corrige a orientação EXIF."""
    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Imagem invalida") from exc

    return fix_exif_orientation(image)


def fix_exif_orientation(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image)


def convert_to_rgb_with_white_background(
    image: Image.Image,
    background_color: tuple[int, int, int] = WHITE,
) -> Image.Image:
    """Converte para RGB e troca transparência por fundo branco."""
    image = fix_exif_orientation(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, background_color + (255,))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    if image.mode != "RGB":
        return image.convert("RGB")
    return image.copy()


def target_size_for_format(target_format: str) -> tuple[int, int]:
    if target_format == FORMAT_15X10_HORIZONTAL:
        return TARGET_15X10_PX
    return TARGET_10X15_PX


def detect_image_orientation(image: Image.Image) -> str:
    """Classifica a orientação da imagem depois da correção de EXIF."""
    width, height = image.size
    if height > width * 1.10:
        return ORIENTATION_VERTICAL
    if width > height * 1.10:
        return ORIENTATION_HORIZONTAL
    return ORIENTATION_SQUARE


def determine_target_format(
    original_orientation: str,
    requested_format: str,
) -> tuple[str, tuple[int, int], str, str | None]:
    """Escolhe o formato final, tamanho em pixels e orientação final."""
    if requested_format != FORMAT_AUTO:
        if requested_format == FORMAT_15X10_HORIZONTAL:
            return FORMAT_15X10_HORIZONTAL, TARGET_15X10_PX, ORIENTATION_HORIZONTAL, None
        return FORMAT_10X15_VERTICAL, TARGET_10X15_PX, ORIENTATION_VERTICAL, None

    if original_orientation == ORIENTATION_HORIZONTAL:
        return FORMAT_15X10_HORIZONTAL, TARGET_15X10_PX, ORIENTATION_HORIZONTAL, None
    if original_orientation == ORIENTATION_SQUARE:
        return (
            FORMAT_10X15_VERTICAL,
            TARGET_10X15_PX,
            ORIENTATION_VERTICAL,
            "Essa foto é quase quadrada; usei 10x15 vertical como padrão.",
        )
    return FORMAT_10X15_VERTICAL, TARGET_10X15_PX, ORIENTATION_VERTICAL, None


def choose_target_format(
    image_size: tuple[int, int],
    requested_format: str,
) -> tuple[str, str | None]:
    probe = Image.new("RGB", image_size)
    orientation = detect_image_orientation(probe)
    selected_format, _, _, warning = determine_target_format(orientation, requested_format)
    return selected_format, warning


def resize_contain(
    image: Image.Image,
    target_size: tuple[int, int],
    background_color: tuple[int, int, int] = WHITE,
) -> Image.Image:
    image = convert_to_rgb_with_white_background(image, background_color)
    target_width, target_height = target_size
    fitted = ImageOps.contain(image, target_size, method=RESAMPLE)
    canvas = Image.new("RGB", target_size, background_color)
    offset_x = (target_width - fitted.width) // 2
    offset_y = (target_height - fitted.height) // 2
    canvas.paste(fitted, (offset_x, offset_y))
    return canvas


def _central_cover_crop_box(
    image_size: tuple[int, int],
    target_size: tuple[int, int],
) -> CropBox:
    image_width, image_height = image_size
    target_width, target_height = target_size
    target_aspect = target_width / target_height
    image_aspect = image_width / image_height

    if image_aspect > target_aspect:
        crop_height = image_height
        crop_width = int(round(crop_height * target_aspect))
    else:
        crop_width = image_width
        crop_height = int(round(crop_width / target_aspect))

    crop_width = max(1, min(crop_width, image_width))
    crop_height = max(1, min(crop_height, image_height))
    crop_x = int(round((image_width - crop_width) / 2))
    crop_y = int(round((image_height - crop_height) / 2))
    return (crop_x, crop_y, crop_width, crop_height)


def calculate_cover_crop_box(
    image_size: tuple[int, int],
    target_aspect: float,
    focus_box: CropBox | None = None,
) -> CropBox:
    """Calcula um crop de cobertura com proporção fixa, opcionalmente focado."""
    image_width, image_height = image_size
    image_aspect = image_width / image_height

    if image_aspect > target_aspect:
        crop_height = image_height
        crop_width = int(round(crop_height * target_aspect))
    else:
        crop_width = image_width
        crop_height = int(round(crop_width / target_aspect))

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


def _crop_and_resize(
    image: Image.Image,
    crop_box: CropBox,
    target_size: tuple[int, int],
) -> Image.Image:
    crop_x, crop_y, crop_width, crop_height = crop_box
    crop_x = max(0, min(crop_x, image.width - 1))
    crop_y = max(0, min(crop_y, image.height - 1))
    crop_width = max(1, min(crop_width, image.width - crop_x))
    crop_height = max(1, min(crop_height, image.height - crop_y))
    cropped = image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
    return cropped.resize(target_size, RESAMPLE).convert("RGB")


def crop_and_resize(
    image: Image.Image,
    crop_box: CropBox | None,
    target_size: tuple[int, int],
    background_color: tuple[int, int, int] = WHITE,
) -> Image.Image:
    if crop_box is None:
        return resize_contain(image, target_size, background_color)
    image = convert_to_rgb_with_white_background(image, background_color)
    return _crop_and_resize(image, crop_box, target_size)


def calculate_crop_requirements(
    image_size: tuple[int, int],
    target_size: tuple[int, int],
) -> CropRequirements:
    image_width, image_height = image_size
    target_width, target_height = target_size
    original_aspect = image_width / image_height
    target_aspect = target_width / target_height

    if abs(original_aspect - target_aspect) < 0.001:
        return CropRequirements("none", 0, 0.0, 0.0, 0.0, 0.0, 0.0, original_aspect, target_aspect)

    if original_aspect > target_aspect:
        crop_width = int(round(image_height * target_aspect))
        removed = max(0, image_width - crop_width)
        percent_width = removed / image_width * 100
        return CropRequirements(
            "width",
            removed,
            removed / 2,
            percent_width,
            percent_width,
            percent_width,
            0.0,
            original_aspect,
            target_aspect,
        )

    crop_height = int(round(image_width / target_aspect))
    removed = max(0, image_height - crop_height)
    percent_height = removed / image_height * 100
    return CropRequirements(
        "height",
        removed,
        removed / 2,
        percent_height,
        percent_height,
        0.0,
        percent_height,
        original_aspect,
        target_aspect,
    )


def classify_crop_amount(crop_percent_total: float, has_faces: bool) -> str:
    if crop_percent_total <= 0.1:
        return "none"
    if crop_percent_total <= 2.0:
        return "tiny"
    if has_faces:
        if crop_percent_total <= 6.0:
            return "small"
        if crop_percent_total <= 8.0:
            return "medium"
        if crop_percent_total <= HIGH_RISK_CROP_PERCENT:
            return "large"
        return "too_large"

    if crop_percent_total <= 8.0:
        return "small"
    if crop_percent_total <= 12.0:
        return "medium"
    if crop_percent_total <= HIGH_RISK_CROP_PERCENT:
        return "large"
    return "too_large"


def get_cropped_regions(image: Image.Image, crop_box: CropBox) -> dict[str, Image.Image]:
    image = convert_to_rgb_with_white_background(image)
    crop_x, crop_y, crop_width, crop_height = crop_box
    image_width, image_height = image.size
    crop_right = crop_x + crop_width
    crop_bottom = crop_y + crop_height
    regions: dict[str, Image.Image] = {}

    if crop_x > 0:
        regions["left"] = image.crop((0, 0, crop_x, image_height))
    if crop_right < image_width:
        regions["right"] = image.crop((crop_right, 0, image_width, image_height))
    if crop_y > 0:
        regions["top"] = image.crop((crop_x, 0, crop_right, crop_y))
    if crop_bottom < image_height:
        regions["bottom"] = image.crop((crop_x, crop_bottom, crop_right, image_height))

    return {
        name: region
        for name, region in regions.items()
        if region.width > 1 and region.height > 1
    }


def calculate_region_importance(region: Image.Image) -> EdgeImportance:
    region = convert_to_rgb_with_white_background(region)
    max_side = max(region.size)
    if max_side > 800:
        scale = 800 / max_side
        resized_size = (max(1, int(region.width * scale)), max(1, int(region.height * scale)))
        region = region.resize(resized_size, RESAMPLE)

    rgb = np.array(region)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / max(1, edges.size))
    brightness_variation = float(np.std(gray) / 128)
    color_variation = float(np.mean(np.std(rgb, axis=(0, 1))) / 128)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_density = min(1.0, len(contours) / max(1.0, edges.size / 2500))
    normalized_std = min(1.0, max(brightness_variation, color_variation))
    importance = min(1.0, edge_density * 5.0 + normalized_std * 0.45 + contour_density * 0.25)

    if importance < 0.20:
        risk_level = "low"
    elif importance <= MAX_EDGE_IMPORTANCE_FOR_SAFE_CROP:
        risk_level = "medium"
    else:
        risk_level = "high"

    return EdgeImportance(
        edge_importance_score=importance,
        edge_density=edge_density,
        color_variation=normalized_std,
        contour_density=contour_density,
        risk_level=risk_level,
    )


def analyze_cropped_areas_importance(image: Image.Image, crop_box: CropBox) -> EdgeImportance:
    regions = get_cropped_regions(image, crop_box)
    if not regions:
        return EdgeImportance(0.0, 0.0, 0.0, 0.0, "low")

    scores = [calculate_region_importance(region) for region in regions.values()]
    return max(scores, key=lambda score: score.edge_importance_score)


def resize_cover_center(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    image = convert_to_rgb_with_white_background(image)
    crop_box = _central_cover_crop_box(image.size, target_size)
    return _crop_and_resize(image, crop_box, target_size)


def _cover_crop_retention(
    image_size: tuple[int, int],
    target_size: tuple[int, int],
) -> float:
    image_width, image_height = image_size
    _, _, crop_width, crop_height = _central_cover_crop_box(image_size, target_size)
    return (crop_width * crop_height) / max(1, image_width * image_height)


def _contain_would_add_borders(
    image_size: tuple[int, int],
    target_size: tuple[int, int],
) -> bool:
    image_ratio = image_size[0] / image_size[1]
    target_ratio = target_size[0] / target_size[1]
    return abs(image_ratio - target_ratio) > 0.01


def _faces_from_analysis_report(report: ImageAnalysisReport | None) -> list[FaceBox]:
    if report is None:
        return []
    faces: list[FaceBox] = []
    for box in report.face_boxes:
        try:
            faces.append((int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])))
        except (KeyError, TypeError, ValueError):
            continue
    return faces


def _local_analysis_contain_reason(
    report: ImageAnalysisReport | None,
    options: ProcessingOptions,
) -> str | None:
    if report is None:
        return None

    if report.required_crop_percent > HIGH_RISK_CROP_PERCENT:
        return "Usei bordas porque o corte necessário seria grande."

    if options.avoid_cutting_people:
        if report.persons_detected > 1 and (report.persons_near_edges or not report.persons_safe_for_crop):
            return "Detectei várias pessoas e preservei o grupo."
        if report.faces_detected > 1 and report.face_group_box:
            _, _, group_width, group_height = report.face_group_box
            group_spread = max(group_width / max(1, report.width), group_height / max(1, report.height))
            if group_spread > 0.55:
                return "Detectei várias pessoas e preservei o grupo."
        if report.faces_near_edges or report.persons_near_edges:
            return "Usei bordas para evitar cortar pessoas."
        if not report.faces_safe_for_crop or not report.persons_safe_for_crop:
            return "Usei bordas para evitar cortar pessoas."

    if options.avoid_cutting_text_or_objects and report.text_near_edges:
        return "Usei bordas porque detectei texto perto da borda."

    if options.avoid_cutting_text_or_objects and report.edge_importance_max > 0.58:
        return "Usei bordas para preservar detalhes importantes perto da borda."

    if (
        options.prefer_borders_when_uncertain
        and report.suggested_strategy in {STRATEGY_CONTAIN, STRATEGY_CREATE_EXTRA_PAGE, STRATEGY_MANUAL_REVIEW}
        and report.risk_level in {"medium", "high"}
    ):
        if report.reasons:
            return report.reasons[0]
        return "Usei bordas porque havia dúvida sobre a segurança do corte."

    return None


def _ai_decision_strategy(ai_decision: AIDecision | None) -> str | None:
    if ai_decision is None:
        return None
    if ai_decision.decision not in AI_ALLOWED_DECISIONS:
        return None
    return ai_decision.decision


def _ai_contain_reason(ai_decision: AIDecision | None) -> str | None:
    strategy = _ai_decision_strategy(ai_decision)
    if strategy not in {"contain_with_borders", "create_extra_page", "manual_review"}:
        return None
    if ai_decision and ai_decision.validation_notes:
        return ai_decision.validation_notes[-1]
    if ai_decision and ai_decision.reason:
        return ai_decision.reason
    return "Usei bordas porque a IA sugeriu a opcao mais conservadora."


def resize_cover_smart(
    image: Image.Image,
    target_size: tuple[int, int],
    faces: list[FaceBox],
    avoid_cutting_people: bool = True,
) -> tuple[Image.Image, bool, str | None, str]:
    image = convert_to_rgb_with_white_background(image)
    if not faces:
        return resize_cover_center(image, target_size), False, None, RESIZE_MODE_LABELS[RESIZE_COVER]

    target_aspect = target_size[0] / target_size[1]
    crop_box = calculate_smart_crop_box(image.size, target_aspect, faces)
    is_safe = is_crop_safe_for_faces(crop_box, faces, image.width, image.height)

    if avoid_cutting_people and not is_safe:
        warning = "Usei bordas para evitar cortar pessoas."
        return (
            resize_contain(image, target_size),
            _contain_would_add_borders(image.size, target_size),
            warning,
            RESIZE_MODE_LABELS[RESIZE_CONTAIN],
        )

    return _crop_and_resize(image, crop_box, target_size), False, None, RESIZE_MODE_LABELS[RESIZE_COVER]


def _safe_crop_score(
    crop_percent_total: float,
    max_allowed_crop_percent: float,
    edge_importance_score: float,
    faces_count: int,
    people_crop_limit: float,
) -> float:
    max_allowed = max(1.0, max_allowed_crop_percent)
    score = 1.0
    score -= min(1.0, crop_percent_total / max_allowed) * 0.40
    score -= edge_importance_score * 0.30
    if faces_count > 1:
        score -= 0.20
    elif faces_count == 1:
        score -= 0.08
    if faces_count and crop_percent_total > people_crop_limit:
        score -= 0.20
    return max(0.0, min(1.0, score))


def decide_safe_crop_or_contain(
    image: Image.Image,
    target_size: tuple[int, int],
    faces: list[FaceBox],
    options: ProcessingOptions,
    analysis_report: ImageAnalysisReport | None = None,
) -> SafeCropDecision:
    """Decide se pode preencher com corte leve ou se deve preservar com bordas."""
    image = convert_to_rgb_with_white_background(image, options.background_color)
    requirements = calculate_crop_requirements(image.size, target_size)
    target_aspect = target_size[0] / target_size[1]
    crop_box = calculate_smart_crop_box(image.size, target_aspect, faces) if faces else calculate_cover_crop_box(image.size, target_aspect)
    crop_class = classify_crop_amount(requirements.crop_percent_total, bool(faces))

    local_contain_reason = _local_analysis_contain_reason(analysis_report, options)
    if local_contain_reason:
        return SafeCropDecision(
            decision=SAFE_CROP_DECISION_CONTAIN,
            reason=local_contain_reason,
            crop_box=None,
            crop_percent_total=requirements.crop_percent_total,
            crop_percent_width=requirements.crop_percent_width,
            crop_percent_height=requirements.crop_percent_height,
            edge_importance_score=analysis_report.edge_importance_max if analysis_report else 1.0,
            safe_crop_score=0.0,
            warning=local_contain_reason,
        )

    if requirements.crop_pixels_total == 0:
        return SafeCropDecision(
            decision=SAFE_CROP_DECISION_CROP,
            reason="A imagem já estava quase no formato certo, então fiz apenas um ajuste leve.",
            crop_box=crop_box,
            crop_percent_total=requirements.crop_percent_total,
            crop_percent_width=requirements.crop_percent_width,
            crop_percent_height=requirements.crop_percent_height,
            edge_importance_score=0.0,
            safe_crop_score=1.0,
            warning=None,
        )

    max_allowed_crop = float(options.max_safe_crop_percent or DEFAULT_MAX_SAFE_CROP_PERCENT)
    people_crop_limit = STRICT_PEOPLE_SAFE_CROP_PERCENT if options.strict_people_safety else max_allowed_crop
    if faces:
        max_allowed_crop = min(max_allowed_crop, people_crop_limit)

    if options.avoid_cutting_people and faces:
        faces_are_safe = is_crop_safe_for_faces(crop_box, faces, image.width, image.height)
        if not faces_are_safe:
            return SafeCropDecision(
                decision=SAFE_CROP_DECISION_CONTAIN,
                reason="Usei bordas para evitar cortar pessoas.",
                crop_box=None,
                crop_percent_total=requirements.crop_percent_total,
                crop_percent_width=requirements.crop_percent_width,
                crop_percent_height=requirements.crop_percent_height,
                edge_importance_score=1.0,
                safe_crop_score=0.0,
                warning="Usei bordas para evitar cortar pessoas.",
            )

    if requirements.crop_percent_total > HIGH_RISK_CROP_PERCENT:
        reason = "Usei bordas porque o corte necessário seria grande."
        return SafeCropDecision(
            decision=SAFE_CROP_DECISION_CONTAIN,
            reason=reason,
            crop_box=None,
            crop_percent_total=requirements.crop_percent_total,
            crop_percent_width=requirements.crop_percent_width,
            crop_percent_height=requirements.crop_percent_height,
            edge_importance_score=1.0,
            safe_crop_score=0.0,
            warning=reason,
        )

    if requirements.crop_percent_total > max_allowed_crop:
        reason = "Usei bordas porque o corte necessário seria maior que o limite escolhido."
        return SafeCropDecision(
            decision=SAFE_CROP_DECISION_CONTAIN,
            reason=reason,
            crop_box=None,
            crop_percent_total=requirements.crop_percent_total,
            crop_percent_width=requirements.crop_percent_width,
            crop_percent_height=requirements.crop_percent_height,
            edge_importance_score=1.0,
            safe_crop_score=0.0,
            warning=reason,
        )

    should_check_edges = (
        options.avoid_cutting_text_or_objects
        and requirements.crop_pixels_total > 0
    )
    if should_check_edges:
        edge_info = analyze_cropped_areas_importance(image, crop_box)
    else:
        edge_info = EdgeImportance(0.0, 0.0, 0.0, 0.0, "low")

    score = _safe_crop_score(
        requirements.crop_percent_total,
        max_allowed_crop,
        edge_info.edge_importance_score,
        len(faces),
        people_crop_limit,
    )

    if options.avoid_cutting_text_or_objects:
        if edge_info.edge_importance_score > 0.55:
            reason = "Usei bordas para preservar detalhes importantes perto da borda."
            return SafeCropDecision(
                decision=SAFE_CROP_DECISION_CONTAIN,
                reason=reason,
                crop_box=None,
                crop_percent_total=requirements.crop_percent_total,
                crop_percent_width=requirements.crop_percent_width,
                crop_percent_height=requirements.crop_percent_height,
                edge_importance_score=edge_info.edge_importance_score,
                safe_crop_score=score,
                warning=reason,
            )
        if edge_info.edge_importance_score > MAX_EDGE_IMPORTANCE_FOR_SAFE_CROP and crop_class not in {"tiny", "small"}:
            reason = "Usei bordas para preservar detalhes importantes perto da borda."
            return SafeCropDecision(
                decision=SAFE_CROP_DECISION_CONTAIN,
                reason=reason,
                crop_box=None,
                crop_percent_total=requirements.crop_percent_total,
                crop_percent_width=requirements.crop_percent_width,
                crop_percent_height=requirements.crop_percent_height,
                edge_importance_score=edge_info.edge_importance_score,
                safe_crop_score=score,
                warning=reason,
            )

    if score >= 0.65 or crop_class == "tiny":
        reason = "Usei corte leve porque as bordas pareciam pouco importantes."
        return SafeCropDecision(
            decision=SAFE_CROP_DECISION_CROP,
            reason=reason,
            crop_box=crop_box,
            crop_percent_total=requirements.crop_percent_total,
            crop_percent_width=requirements.crop_percent_width,
            crop_percent_height=requirements.crop_percent_height,
            edge_importance_score=edge_info.edge_importance_score,
            safe_crop_score=score,
            warning=None,
        )

    if score >= 0.50 and not options.prefer_borders_when_uncertain:
        reason = "Corte leve aplicado para preencher sem bordas."
        return SafeCropDecision(
            decision=SAFE_CROP_DECISION_CROP,
            reason=reason,
            crop_box=crop_box,
            crop_percent_total=requirements.crop_percent_total,
            crop_percent_width=requirements.crop_percent_width,
            crop_percent_height=requirements.crop_percent_height,
            edge_importance_score=edge_info.edge_importance_score,
            safe_crop_score=score,
            warning=None,
        )

    reason = "Usei bordas porque havia dúvida sobre a segurança do corte."
    return SafeCropDecision(
        decision=SAFE_CROP_DECISION_CONTAIN,
        reason=reason,
        crop_box=None,
        crop_percent_total=requirements.crop_percent_total,
        crop_percent_width=requirements.crop_percent_width,
        crop_percent_height=requirements.crop_percent_height,
        edge_importance_score=edge_info.edge_importance_score,
        safe_crop_score=score,
        warning=reason,
    )


def _low_resolution_warning(
    image_size: tuple[int, int],
    target_size: tuple[int, int],
) -> str | None:
    source_width, source_height = image_size
    target_width, target_height = target_size
    if source_width < target_width * 0.70 or source_height < target_height * 0.70:
        return "Essa foto tem resolução baixa e pode perder qualidade na impressão."
    return None


def process_image(
    image: Image.Image,
    original_name: str,
    index: int,
    options: ProcessingOptions,
) -> ProcessedImage:
    """Processa uma foto individual mantendo sempre a proporção original."""
    rgb_image = convert_to_rgb_with_white_background(image, options.background_color)
    original_orientation = detect_image_orientation(rgb_image)
    selected_format, target_size, final_orientation, format_warning = determine_target_format(
        original_orientation,
        options.target_format,
    )

    analysis_report: ImageAnalysisReport | None = None
    if options.use_local_analysis:
        try:
            analysis_report = analyze_image_locally(
                rgb_image,
                original_name,
                TARGET_LABELS[selected_format],
                target_size,
                frame_priority=options.frame_priority,
            )
        except Exception:
            logger.info("Analise local indisponivel; seguindo com processamento padrao")

    if analysis_report is not None:
        faces = _faces_from_analysis_report(analysis_report)
    else:
        faces = detect_faces(rgb_image)

    ai_decision: AIDecision | None = None
    ai_report_payload: dict | None = None
    ai_prompt: str | None = None
    ai_raw_response_text: str | None = None
    ai_error: str | None = None
    if options.use_ai_decision and analysis_report is not None:
        try:
            ai_result = get_ai_decision_for_report(
                analysis_report,
                max_safe_crop_percent=float(options.max_safe_crop_percent or DEFAULT_MAX_SAFE_CROP_PERCENT),
            )
            ai_decision = ai_result.decision
            ai_report_payload = ai_result.report_payload
            ai_prompt = ai_result.prompt
            ai_raw_response_text = ai_result.raw_response_text
            ai_error = ai_result.error
        except Exception:
            logger.info("IA de decisao indisponivel; seguindo com analise local")
            ai_error = "IA indisponivel"

    final_decision_strategy = _ai_decision_strategy(ai_decision)
    if final_decision_strategy is None and analysis_report is not None:
        final_decision_strategy = analysis_report.suggested_strategy
    local_crop_override_reason = None
    if final_decision_strategy in {STRATEGY_SAFE_CROP, STRATEGY_SMART_FACE_CROP, STRATEGY_CENTER_CROP}:
        local_crop_override_reason = _local_analysis_contain_reason(analysis_report, options)
        if local_crop_override_reason:
            final_decision_strategy = STRATEGY_CONTAIN

    warnings: list[str | None] = [format_warning, _low_resolution_warning(rgb_image.size, target_size)]
    if options.use_ai_decision and ai_error:
        warnings.append("A IA nao respondeu, entao usei a analise local.")
    used_borders = False
    used_smart_crop = False
    used_safe_crop = False
    used_subject_focused_crop = False
    subject_crop_reason: str | None = None
    safe_crop_decision: SafeCropDecision | None = None
    mode_used = RESIZE_MODE_LABELS[options.resize_mode]

    if options.resize_mode == RESIZE_CONTAIN:
        processed = resize_contain(rgb_image, target_size, options.background_color)
        used_borders = _contain_would_add_borders(rgb_image.size, target_size)
    elif options.resize_mode == RESIZE_COVER:
        local_contain_reason = _ai_contain_reason(ai_decision) or _local_analysis_contain_reason(analysis_report, options)
        if local_contain_reason:
            processed = resize_contain(rgb_image, target_size, options.background_color)
            used_borders = _contain_would_add_borders(rgb_image.size, target_size)
            mode_used = RESIZE_MODE_LABELS[RESIZE_CONTAIN]
            warnings.append(local_contain_reason)
        else:
            processed, used_borders, smart_warning, mode_used = resize_cover_smart(
                rgb_image,
                target_size,
                faces,
                avoid_cutting_people=options.avoid_cutting_people,
            )
            warnings.append(smart_warning)
            used_smart_crop = bool(faces and not used_borders and mode_used == RESIZE_MODE_LABELS[RESIZE_COVER])
    elif options.resize_mode in (RESIZE_SAFE_CROP, RESIZE_SMART):
        if options.safe_crop_enabled:
            ai_contain_reason = _ai_contain_reason(ai_decision) or local_crop_override_reason
            if ai_contain_reason:
                safe_crop_decision = SafeCropDecision(
                    decision=SAFE_CROP_DECISION_CONTAIN,
                    reason=ai_contain_reason,
                    crop_box=None,
                    crop_percent_total=analysis_report.required_crop_percent if analysis_report else 0.0,
                    crop_percent_width=analysis_report.required_crop_percent if analysis_report and analysis_report.required_crop_axis == "width" else 0.0,
                    crop_percent_height=analysis_report.required_crop_percent if analysis_report and analysis_report.required_crop_axis == "height" else 0.0,
                    edge_importance_score=analysis_report.edge_importance_max if analysis_report else 0.0,
                    safe_crop_score=0.0,
                    warning=ai_contain_reason,
                )
                processed = resize_contain(rgb_image, target_size, options.background_color)
                used_borders = _contain_would_add_borders(rgb_image.size, target_size)
                mode_used = RESIZE_MODE_LABELS[RESIZE_CONTAIN]
                warnings.append(safe_crop_decision.warning)
            elif (
                analysis_report is not None
                and (
                    final_decision_strategy == STRATEGY_SUBJECT_FOCUSED_CROP
                    or (
                        options.frame_priority != FRAME_PRIORITY_KEEP_SCENE
                        and analysis_report.can_tighten_frame
                    )
                )
            ):
                subject_decision = decide_subject_crop_strategy(
                    rgb_image,
                    analysis_report,
                    analysis_report.face_boxes,
                    analysis_report.person_boxes,
                    analysis_report.object_boxes,
                    analysis_report.text_boxes,
                    target_size,
                    options.frame_priority,
                )
                analysis_report.primary_subject_type = subject_decision.primary_subject_type
                analysis_report.primary_subject_box = subject_decision.primary_subject_box
                analysis_report.primary_subject_expanded_box = subject_decision.expanded_subject_box
                analysis_report.primary_subject_confidence = subject_decision.primary_subject_confidence
                analysis_report.subject_focus_score = subject_decision.subject_focus_score
                analysis_report.background_waste_score = subject_decision.background_waste_score
                analysis_report.empty_area_top_score = subject_decision.empty_area_scores.get("top", 0.0)
                analysis_report.empty_area_bottom_score = subject_decision.empty_area_scores.get("bottom", 0.0)
                analysis_report.empty_area_left_score = subject_decision.empty_area_scores.get("left", 0.0)
                analysis_report.empty_area_right_score = subject_decision.empty_area_scores.get("right", 0.0)
                analysis_report.can_tighten_frame = subject_decision.can_tighten_frame
                analysis_report.recommended_crop_mode = subject_decision.strategy
                analysis_report.subject_crop_reason = subject_decision.reason

                if subject_decision.strategy == STRATEGY_SUBJECT_FOCUSED_CROP and subject_decision.crop_box:
                    processed = apply_subject_focused_crop(rgb_image, subject_decision.crop_box, target_size)
                    used_safe_crop = True
                    used_subject_focused_crop = True
                    subject_crop_reason = subject_decision.reason
                    mode_used = "Foco nas pessoas"
                    final_decision_strategy = STRATEGY_SUBJECT_FOCUSED_CROP
                    safe_crop_decision = SafeCropDecision(
                        decision=SAFE_CROP_DECISION_CROP,
                        reason=subject_decision.reason,
                        crop_box=subject_decision.crop_box,
                        crop_percent_total=analysis_report.required_crop_percent,
                        crop_percent_width=analysis_report.required_crop_percent if analysis_report.required_crop_axis == "width" else 0.0,
                        crop_percent_height=analysis_report.required_crop_percent if analysis_report.required_crop_axis == "height" else 0.0,
                        edge_importance_score=analysis_report.edge_importance_max,
                        safe_crop_score=subject_decision.subject_focus_score,
                        warning=None,
                    )
                    warnings.append(subject_decision.reason)
                elif (
                    final_decision_strategy == STRATEGY_SUBJECT_FOCUSED_CROP
                    and subject_decision.strategy == STRATEGY_CONTAIN
                ):
                    subject_crop_reason = subject_decision.reason
                    processed = resize_contain(rgb_image, target_size, options.background_color)
                    used_borders = _contain_would_add_borders(rgb_image.size, target_size)
                    mode_used = RESIZE_MODE_LABELS[RESIZE_CONTAIN]
                    final_decision_strategy = STRATEGY_CONTAIN
                    safe_crop_decision = SafeCropDecision(
                        decision=SAFE_CROP_DECISION_CONTAIN,
                        reason=subject_decision.reason,
                        crop_box=None,
                        crop_percent_total=analysis_report.required_crop_percent,
                        crop_percent_width=analysis_report.required_crop_percent if analysis_report.required_crop_axis == "width" else 0.0,
                        crop_percent_height=analysis_report.required_crop_percent if analysis_report.required_crop_axis == "height" else 0.0,
                        edge_importance_score=analysis_report.edge_importance_max,
                        safe_crop_score=0.0,
                        warning=subject_decision.reason,
                    )
                    warnings.append(subject_decision.reason)
                else:
                    safe_crop_decision = decide_safe_crop_or_contain(
                        rgb_image,
                        target_size,
                        faces,
                        options,
                        analysis_report,
                    )
                    if safe_crop_decision.decision == SAFE_CROP_DECISION_CROP:
                        processed = crop_and_resize(
                            rgb_image,
                            safe_crop_decision.crop_box,
                            target_size,
                            options.background_color,
                        )
                        used_safe_crop = True
                        used_smart_crop = bool(faces)
                        mode_used = RESIZE_MODE_LABELS[RESIZE_SAFE_CROP]
                    else:
                        processed = resize_contain(rgb_image, target_size, options.background_color)
                        used_borders = _contain_would_add_borders(rgb_image.size, target_size)
                        mode_used = RESIZE_MODE_LABELS[RESIZE_CONTAIN]
                        warnings.append(safe_crop_decision.warning)
            elif final_decision_strategy == STRATEGY_CENTER_CROP:
                processed = resize_cover_center(rgb_image, target_size)
                used_safe_crop = True
                mode_used = RESIZE_MODE_LABELS[RESIZE_SAFE_CROP]
            elif final_decision_strategy == STRATEGY_SMART_FACE_CROP and faces:
                processed, used_borders, smart_warning, mode_used = resize_cover_smart(
                    rgb_image,
                    target_size,
                    faces,
                    avoid_cutting_people=options.avoid_cutting_people,
                )
                warnings.append(smart_warning)
                used_smart_crop = not used_borders
            else:
                safe_crop_decision = decide_safe_crop_or_contain(
                    rgb_image,
                    target_size,
                    faces,
                    options,
                    analysis_report,
                )
                if safe_crop_decision.decision == SAFE_CROP_DECISION_CROP:
                    processed = crop_and_resize(
                        rgb_image,
                        safe_crop_decision.crop_box,
                        target_size,
                        options.background_color,
                    )
                    used_safe_crop = True
                    used_smart_crop = bool(faces)
                    mode_used = RESIZE_MODE_LABELS[RESIZE_SAFE_CROP]
                else:
                    processed = resize_contain(rgb_image, target_size, options.background_color)
                    used_borders = _contain_would_add_borders(rgb_image.size, target_size)
                    mode_used = RESIZE_MODE_LABELS[RESIZE_CONTAIN]
                    warnings.append(safe_crop_decision.warning)
        else:
            processed = resize_contain(rgb_image, target_size, options.background_color)
            used_borders = _contain_would_add_borders(rgb_image.size, target_size)
            mode_used = RESIZE_MODE_LABELS[RESIZE_CONTAIN]
    else:
        processed = resize_contain(rgb_image, target_size, options.background_color)
        used_borders = _contain_would_add_borders(rgb_image.size, target_size)
        mode_used = RESIZE_MODE_LABELS[RESIZE_CONTAIN]

    format_suffix = "15x10" if selected_format == FORMAT_15X10_HORIZONTAL else "10x15"
    output_name = f"foto_{index:03d}_{sanitize_filename(original_orientation)}_{format_suffix}.jpg"

    return ProcessedImage(
        index=index,
        original_name=Path(original_name).name,
        output_name=output_name,
        image=processed,
        target_format=TARGET_LABELS[selected_format],
        resize_mode_used=mode_used,
        faces_detected=len(faces),
        used_borders=used_borders,
        source_image=rgb_image.copy(),
        original_image=rgb_image.copy(),
        warning=merge_warnings(warnings),
        original_size=rgb_image.size,
        original_width=rgb_image.width,
        original_height=rgb_image.height,
        original_orientation=original_orientation,
        final_orientation=final_orientation,
        target_size_px=target_size,
        resize_mode_requested=RESIZE_MODE_LABELS[options.resize_mode],
        multiple_faces_detected=len(faces) > 1,
        used_smart_crop=used_smart_crop,
        safe_crop_decision=safe_crop_decision.decision if safe_crop_decision else None,
        safe_crop_score=safe_crop_decision.safe_crop_score if safe_crop_decision else None,
        crop_percent_width=safe_crop_decision.crop_percent_width if safe_crop_decision else None,
        crop_percent_height=safe_crop_decision.crop_percent_height if safe_crop_decision else None,
        crop_percent_total=safe_crop_decision.crop_percent_total if safe_crop_decision else None,
        edge_importance_score=safe_crop_decision.edge_importance_score if safe_crop_decision else None,
        safe_crop_reason=safe_crop_decision.reason if safe_crop_decision else None,
        used_safe_crop=used_safe_crop,
        analysis_report=analysis_report,
        ai_decision=ai_decision,
        ai_report_payload=ai_report_payload,
        ai_prompt=ai_prompt,
        ai_raw_response_text=ai_raw_response_text,
        ai_error=ai_error,
        final_decision_strategy=final_decision_strategy,
        ai_rotate_on_pdf_requested=bool(ai_decision and ai_decision.rotate_on_pdf),
        ai_create_extra_page_requested=bool(ai_decision and ai_decision.create_extra_page),
        used_subject_focused_crop=used_subject_focused_crop,
        subject_crop_reason=subject_crop_reason,
    )


def save_jpg_bytes(
    image: Image.Image,
    dpi: tuple[int, int] = (DPI, DPI),
    quality: int = 95,
) -> BytesIO:
    output = BytesIO()
    image.convert("RGB").save(
        output,
        format="JPEG",
        quality=quality,
        optimize=True,
        dpi=dpi,
    )
    output.seek(0)
    return output


def generate_preview_image(image: Image.Image, max_size: tuple[int, int] = (480, 480)) -> Image.Image:
    preview = image.copy()
    preview.thumbnail(max_size, RESAMPLE)
    return preview
