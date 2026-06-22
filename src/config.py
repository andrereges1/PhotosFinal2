"""Constantes compartilhadas do Foto 10x15 Facil."""

from __future__ import annotations

import os

DPI = 300
CM_PER_INCH = 2.54

A4_WIDTH_CM = 21.0
A4_HEIGHT_CM = 29.7
PAGE_MARGIN_CM = 1.0
DEFAULT_GAP_CM = 0.4

PDF_EDGE_MARGIN_CM = 0.3
PDF_PHOTO_GAP_CM = 0.2

PHOTO_4UP_WIDTH_CM = 10.0
PHOTO_4UP_HEIGHT_CM = 14.52
PDF_4UP_COLUMNS = 2
PDF_4UP_ROWS = 2
PDF_4UP_HORIZONTAL_GAP_CM = 0.4
PDF_4UP_VERTICAL_GAP_CM = 0.06

PHOTO_REAL_SHORT_CM = 10.0
PHOTO_REAL_LONG_CM = 15.0

PHOTO_VERTICAL_WIDTH_CM = 10.0
PHOTO_VERTICAL_HEIGHT_CM = 15.0

PHOTO_HORIZONTAL_WIDTH_CM = 15.0
PHOTO_HORIZONTAL_HEIGHT_CM = 10.0

PHOTO_10X15_WIDTH_CM = 10.0
PHOTO_10X15_HEIGHT_CM = 15.0

TARGET_10X15_PX = (1181, 1772)
TARGET_15X10_PX = (1772, 1181)
TARGET_10X1452_PX = (1181, 1715)

WHITE = (255, 255, 255)

ORIENTATION_VERTICAL = "vertical"
ORIENTATION_HORIZONTAL = "horizontal"
ORIENTATION_SQUARE = "quadrada"

FORMAT_AUTO = "automatico"
FORMAT_10X15_VERTICAL = "10x15_vertical"
FORMAT_15X10_HORIZONTAL = "15x10_horizontal"

RESIZE_SMART = "automatico_inteligente"
RESIZE_SAFE_CROP = "preencher_com_corte_seguro"
RESIZE_CONTAIN = "manter_inteira_com_bordas"
RESIZE_COVER = "preencher_e_cortar"

FRAME_PRIORITY_KEEP_SCENE = "Preservar mais cenario"
FRAME_PRIORITY_BALANCED = "Equilibrado"
FRAME_PRIORITY_PEOPLE = "Foco nas pessoas"

STRATEGY_SUBJECT_FOCUSED_CROP = "subject_focused_crop"
STRATEGY_SAFE_CROP = "safe_crop"
STRATEGY_CONTAIN_BORDERS = "contain_with_borders"
STRATEGY_CENTER_CROP = "center_crop"
STRATEGY_SMART_FACE_CROP = "smart_face_crop"

PRIMARY_SUBJECT_SINGLE_PERSON = "single_person"
PRIMARY_SUBJECT_GROUP_PEOPLE = "group_people"
PRIMARY_SUBJECT_PEOPLE_WITH_PET = "people_with_pet"
PRIMARY_SUBJECT_OBJECT = "object"
PRIMARY_SUBJECT_MIXED_SCENE = "mixed_scene"
PRIMARY_SUBJECT_UNKNOWN = "unknown"

DEFAULT_SUBJECT_MARGIN_LEFT = 0.10
DEFAULT_SUBJECT_MARGIN_RIGHT = 0.10
DEFAULT_SUBJECT_MARGIN_TOP = 0.12
DEFAULT_SUBJECT_MARGIN_BOTTOM = 0.10

PEOPLE_FOCUS_MARGIN_LEFT = 0.07
PEOPLE_FOCUS_MARGIN_RIGHT = 0.07
PEOPLE_FOCUS_MARGIN_TOP = 0.09
PEOPLE_FOCUS_MARGIN_BOTTOM = 0.08

KEEP_SCENE_MARGIN_LEFT = 0.16
KEEP_SCENE_MARGIN_RIGHT = 0.16
KEEP_SCENE_MARGIN_TOP = 0.18
KEEP_SCENE_MARGIN_BOTTOM = 0.16

MAX_SUBJECT_CROP_PERCENT_BALANCED = 28.0
MAX_SUBJECT_CROP_PERCENT_PEOPLE = 38.0
MAX_SUBJECT_CROP_PERCENT_KEEP_SCENE = 18.0

MIN_SUBJECT_COVERAGE_RATIO = 0.55
MAX_EMPTY_BACKGROUND_SCORE_FOR_CROP = 0.70

DEFAULT_MAX_SAFE_CROP_PERCENT = 12.0
STRICT_PEOPLE_SAFE_CROP_PERCENT = 8.0
MAX_EDGE_IMPORTANCE_FOR_SAFE_CROP = 0.35
MIN_CROP_PERCENT_TO_CARE = 2.0
HIGH_RISK_CROP_PERCENT = 20.0

SAFE_CROP_DECISION_CROP = "crop"
SAFE_CROP_DECISION_CONTAIN = "contain"
SAFE_CROP_DECISION_WARNING = "warning"

PDF_LAYOUT_3_PER_A4 = "3_fotos_por_a4_proporcao"
PDF_LAYOUT_3_PROPORTION = PDF_LAYOUT_3_PER_A4
PDF_LAYOUT_3_REAL_PHOTOS = "3 fotos 10x15 reais por A4"
PDF_LAYOUT_4_REAL_PHOTOS = "4 imagens 10x14,52 por A4"
PDF_LAYOUT_2_REAL = "2_fotos_10x15_reais"

PDF_ORGANIZE_AUTO = "organizar_automaticamente"
PDF_ORGANIZE_UPLOAD_ORDER = "manter_ordem_upload"

PRIORITY_SAFE = "safe_first"
PRIORITY_BALANCED = "balanced"
PRIORITY_PAPER_SAVING = "paper_saving"
PRIORITY_FILL_PHOTO = "fill_photo"

BATCH_STRATEGY_PRESERVE_ORDER = "preserve_order"
BATCH_STRATEGY_BEST_FIT = "best_fit"
BATCH_STRATEGY_SAFE_FIRST = "safe_first"
BATCH_STRATEGY_PAPER_SAVING = "paper_saving"
BATCH_STRATEGY_MIXED = "mixed"

BATCH_PLAN_SOURCES = {"local", "ai", "ai_validated", "fallback"}
BATCH_ALLOWED_STRATEGIES = {
    BATCH_STRATEGY_PRESERVE_ORDER,
    BATCH_STRATEGY_BEST_FIT,
    BATCH_STRATEGY_SAFE_FIRST,
    BATCH_STRATEGY_PAPER_SAVING,
    BATCH_STRATEGY_MIXED,
}
BATCH_ALLOWED_LAYOUT_TYPES = {
    "3_real_photos_a4",
    "4_real_images_a4",
    "3_photos_mixed",
    "2_horizontal",
    "2_real_size",
    "single_photo",
    "custom_safe",
}
BATCH_ALLOWED_SLOT_POSITIONS = {
    "top",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "center",
    "top_1",
    "top_2",
}
BATCH_ALLOWED_FIT_STRATEGIES = {
    "normal",
    "safe_crop",
    "contain_with_borders",
    "rotate_on_pdf",
    "extra_page",
}
BATCH_ALLOWED_FINAL_STRATEGIES = {
    "safe_crop",
    "subject_focused_crop",
    "contain_with_borders",
    "smart_face_crop",
    "center_crop",
    "rotate_on_pdf",
    "create_extra_page",
    "manual_review_fallback",
}

TARGET_LABELS = {
    FORMAT_AUTO: "Automático",
    FORMAT_10X15_VERTICAL: "10x15 vertical",
    FORMAT_15X10_HORIZONTAL: "15x10 horizontal",
}

RESIZE_MODE_LABELS = {
    RESIZE_SMART: "Automático inteligente",
    RESIZE_SAFE_CROP: "Preencher com corte seguro",
    RESIZE_CONTAIN: "Manter foto inteira com bordas",
    RESIZE_COVER: "Preencher e cortar o mínimo necessário",
}

PDF_LAYOUT_LABELS = {
    PDF_LAYOUT_3_REAL_PHOTOS: "3 fotos 10x15 reais por A4",
    PDF_LAYOUT_4_REAL_PHOTOS: "4 imagens 10x14,52 por A4",
    PDF_LAYOUT_3_PER_A4: "3 fotos em proporção 10x15",
    PDF_LAYOUT_2_REAL: "2 fotos 10x15 reais",
}

def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    _load_dotenv_if_available()
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "sim"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "nao", "não"}:
        return False
    return default


AI_DECISION_ENABLED_DEFAULT = _env_bool("AI_DECISION_ENABLED", False)
AI_PROVIDER_NAME = "opencode_zen"
AI_DEFAULT_BASE_URL = "https://opencode.ai/zen"
AI_DEFAULT_MODEL = "minimax-m3-free"
AI_DEFAULT_ENDPOINT_PATH = "/v1/chat/completions"
AI_REQUEST_TIMEOUT_SECONDS = 60
AI_MAX_RETRIES = 1
AI_STRICT_JSON_MODE = True
AI_ALLOW_IMAGE_DATA = False
AI_ALLOWED_DECISIONS = [
    "safe_crop",
    "subject_focused_crop",
    "contain_with_borders",
    "smart_face_crop",
    "center_crop",
    "rotate_on_pdf",
    "create_extra_page",
    "manual_review",
]
AI_FALLBACK_DECISION = "contain_with_borders"


def cm_to_points(cm: float) -> float:
    """Converte centímetros para points, unidade usada pelo ReportLab."""
    return cm * 72 / CM_PER_INCH


def points_to_cm(points: float) -> float:
    """Converte points do PDF para centímetros."""
    return points * CM_PER_INCH / 72


def cm_to_pixels(cm: float, dpi: int = DPI) -> int:
    """Converte centímetros para pixels no DPI informado."""
    return round(cm / CM_PER_INCH * dpi)


def pixels_to_cm(px: int, dpi: int = DPI) -> float:
    """Converte pixels para centímetros no DPI informado."""
    return px / dpi * CM_PER_INCH
