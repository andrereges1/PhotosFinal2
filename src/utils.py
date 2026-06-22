"""Tipos e utilidades pequenas para o app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Iterator, Sequence

from PIL import Image

from src.config import (
    DEFAULT_GAP_CM,
    DEFAULT_MAX_SAFE_CROP_PERCENT,
    DPI,
    PDF_ORGANIZE_AUTO,
    PAGE_MARGIN_CM,
    PDF_LAYOUT_3_REAL_PHOTOS,
    PDF_LAYOUT_3_PER_A4,
    RESIZE_SMART,
    WHITE,
    FORMAT_AUTO,
    FRAME_PRIORITY_BALANCED,
)


@dataclass(slots=True)
class ProcessingOptions:
    target_format: str = FORMAT_AUTO
    resize_mode: str = RESIZE_SMART
    avoid_cutting_people: bool = True
    background_color: tuple[int, int, int] = WHITE
    dpi: int = DPI
    max_safe_crop_percent: float = DEFAULT_MAX_SAFE_CROP_PERCENT
    prefer_borders_when_uncertain: bool = True
    avoid_cutting_text_or_objects: bool = True
    strict_people_safety: bool = True
    safe_crop_enabled: bool = True
    use_local_analysis: bool = True
    use_ai_decision: bool = False
    frame_priority: str = FRAME_PRIORITY_BALANCED


@dataclass(slots=True)
class PdfOptions:
    layout_mode: str = PDF_LAYOUT_3_REAL_PHOTOS
    show_cut_lines: bool = True
    margin_cm: float = PAGE_MARGIN_CM
    gap_cm: float = DEFAULT_GAP_CM
    page_orientation: str = "auto"
    organize_mode: str = PDF_ORGANIZE_AUTO


@dataclass(slots=True)
class ProcessedImage:
    index: int
    original_name: str
    output_name: str
    image: Image.Image
    target_format: str
    resize_mode_used: str
    faces_detected: int
    used_borders: bool
    source_image: Image.Image | None = None
    original_image: Image.Image | None = None
    warning: str | None = None
    original_size: tuple[int, int] | None = None
    original_width: int = 0
    original_height: int = 0
    original_orientation: str = ""
    final_orientation: str = ""
    target_size_px: tuple[int, int] = (0, 0)
    resize_mode_requested: str = ""
    multiple_faces_detected: bool = False
    used_smart_crop: bool = False
    pdf_slot_type: str | None = None
    rotated_on_pdf: bool = False
    pdf_rotation_degrees: int = 0
    pdf_rotation_reason: str | None = None
    pdf_layout_name: str | None = None
    pdf_page_number: int | None = None
    pdf_position_label: str | None = None
    safe_crop_decision: str | None = None
    safe_crop_score: float | None = None
    crop_percent_width: float | None = None
    crop_percent_height: float | None = None
    crop_percent_total: float | None = None
    edge_importance_score: float | None = None
    safe_crop_reason: str | None = None
    used_safe_crop: bool = False
    analysis_report: Any | None = None
    ai_decision: Any | None = None
    ai_report_payload: dict[str, Any] | None = None
    ai_prompt: str | None = None
    ai_raw_response_text: str | None = None
    ai_error: str | None = None
    final_decision_strategy: str | None = None
    ai_rotate_on_pdf_requested: bool = False
    ai_create_extra_page_requested: bool = False
    batch_final_strategy: str | None = None
    batch_plan_source: str | None = None
    batch_validation_notes: list[str] | None = None
    batch_pdf_slot_position: str | None = None
    used_subject_focused_crop: bool = False
    subject_crop_reason: str | None = None


def ensure_output_dirs(base_dir: Path | str = "output") -> None:
    base = Path(base_dir)
    (base / "imagens").mkdir(parents=True, exist_ok=True)
    (base / "pdf").mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Gera um nome simples e seguro para arquivos baixados."""
    stem = Path(name).stem or "foto"
    normalized = unicodedata.normalize("NFKD", stem)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", ascii_name).strip("._")
    return clean or "foto"


def chunked(items: Sequence[ProcessedImage], size: int) -> Iterator[list[ProcessedImage]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def merge_warnings(warnings: Iterable[str | None]) -> str | None:
    clean = [warning.strip() for warning in warnings if warning and warning.strip()]
    if not clean:
        return None
    unique = list(dict.fromkeys(clean))
    return " ".join(unique)
