"""Planejamento local e payload tecnico do lote inteiro."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any

from PIL import Image

from src.analysis_models import BatchPlan, BatchReport, FinalImageDecision, PlannedPage, PlannedSlot
from src.config import (
    AI_FALLBACK_DECISION,
    BATCH_STRATEGY_BEST_FIT,
    BATCH_STRATEGY_MIXED,
    BATCH_STRATEGY_PAPER_SAVING,
    BATCH_STRATEGY_PRESERVE_ORDER,
    BATCH_STRATEGY_SAFE_FIRST,
    DEFAULT_MAX_SAFE_CROP_PERCENT,
    MAX_EDGE_IMPORTANCE_FOR_SAFE_CROP,
    ORIENTATION_HORIZONTAL,
    ORIENTATION_SQUARE,
    ORIENTATION_VERTICAL,
    PDF_ORGANIZE_UPLOAD_ORDER,
    PDF_LAYOUT_2_REAL,
    PDF_LAYOUT_3_REAL_PHOTOS,
    PDF_LAYOUT_4_REAL_PHOTOS,
    PRIORITY_FILL_PHOTO,
    PRIORITY_PAPER_SAVING,
    PRIORITY_SAFE,
)
from src.utils import ProcessedImage

logger = logging.getLogger(__name__)

FORBIDDEN_AI_PAYLOAD_KEYS = {
    "image",
    "image_bytes",
    "base64",
    "pixels",
    "raw_data",
    "file_content",
    "pil_image",
}


def _round(value: Any, digits: int = 4) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _risk_level(item: ProcessedImage) -> str:
    report = item.analysis_report
    if report and report.risk_level in {"low", "medium", "high"}:
        return report.risk_level
    return "medium"


def _required_crop_percent(item: ProcessedImage) -> float:
    report = item.analysis_report
    if report:
        return _round(report.required_crop_percent, 2)
    return _round(item.crop_percent_total, 2)


def _edge_importance_max(item: ProcessedImage) -> float:
    report = item.analysis_report
    if report:
        return _round(report.edge_importance_max)
    return _round(item.edge_importance_score)


def _is_horizontal(item: ProcessedImage) -> bool:
    if item.final_orientation:
        return item.final_orientation == ORIENTATION_HORIZONTAL
    return item.image.width >= item.image.height


def _is_high_or_unsafe(item: ProcessedImage) -> bool:
    report = item.analysis_report
    if report is None:
        return False
    return (
        report.risk_level == "high"
        or report.faces_near_edges
        or report.persons_near_edges
        or (report.faces_detected > 0 and not report.faces_safe_for_crop)
        or (report.persons_detected > 0 and not report.persons_safe_for_crop)
        or (report.text_detected and report.text_near_edges)
        or report.edge_importance_max > 0.58
    )


def _compact_image_report(item: ProcessedImage) -> dict[str, Any]:
    report = item.analysis_report
    return {
        "id": f"foto_{item.index:03d}",
        "image_name": item.output_name,
        "orientation": item.original_orientation,
        "final_orientation": item.final_orientation,
        "risk_level": _risk_level(item),
        "required_crop_percent": _required_crop_percent(item),
        "faces_detected": int(getattr(report, "faces_detected", item.faces_detected) if report else item.faces_detected),
        "persons_detected": int(getattr(report, "persons_detected", 0) if report else 0),
        "text_near_edges": bool(getattr(report, "text_near_edges", False) if report else False),
        "edge_importance_max": _edge_importance_max(item),
        "local_suggestion": getattr(report, "suggested_strategy", None) or item.final_decision_strategy or AI_FALLBACK_DECISION,
        "primary_subject_type": getattr(report, "primary_subject_type", "unknown") if report else "unknown",
        "subject_focus_score": _round(getattr(report, "subject_focus_score", 0.0) if report else 0.0),
        "background_waste_score": _round(getattr(report, "background_waste_score", 0.0) if report else 0.0),
        "can_tighten_frame": bool(getattr(report, "can_tighten_frame", False) if report else False),
        "recommended_crop_mode": getattr(report, "recommended_crop_mode", None) or AI_FALLBACK_DECISION,
    }


def _suggest_batch_strategy(processed_images: list[ProcessedImage], user_preferences: dict[str, Any]) -> str:
    priority = user_preferences.get("priority_mode")
    preserve_order = bool(user_preferences.get("preserve_upload_order"))
    high_risk_count = sum(1 for item in processed_images if _risk_level(item) == "high")
    if preserve_order:
        return BATCH_STRATEGY_PRESERVE_ORDER
    if priority == PRIORITY_SAFE or high_risk_count:
        return BATCH_STRATEGY_SAFE_FIRST
    if priority == PRIORITY_PAPER_SAVING:
        return BATCH_STRATEGY_PAPER_SAVING
    if priority == PRIORITY_FILL_PHOTO:
        return BATCH_STRATEGY_MIXED
    return BATCH_STRATEGY_BEST_FIT


def build_batch_report(
    processed_images: list[ProcessedImage],
    user_preferences: dict[str, Any],
) -> BatchReport:
    vertical_count = sum(1 for item in processed_images if item.original_orientation == ORIENTATION_VERTICAL)
    horizontal_count = sum(1 for item in processed_images if item.original_orientation == ORIENTATION_HORIZONTAL)
    square_count = sum(1 for item in processed_images if item.original_orientation == ORIENTATION_SQUARE)
    low_risk_count = sum(1 for item in processed_images if _risk_level(item) == "low")
    medium_risk_count = sum(1 for item in processed_images if _risk_level(item) == "medium")
    high_risk_count = sum(1 for item in processed_images if _risk_level(item) == "high")
    faces_total = sum(int(getattr(item.analysis_report, "faces_detected", item.faces_detected)) for item in processed_images)
    persons_total = sum(int(getattr(item.analysis_report, "persons_detected", 0)) for item in processed_images)
    text_images_count = sum(1 for item in processed_images if bool(getattr(item.analysis_report, "text_detected", False)))
    warnings: list[str] = []
    if high_risk_count:
        warnings.append("Ha fotos de alto risco; o planejamento local prioriza bordas e preservacao.")

    used_borders_count = sum(1 for item in processed_images if item.used_borders)
    safe_crop_count = sum(1 for item in processed_images if item.used_safe_crop and not item.used_subject_focused_crop)
    subject_crop_count = sum(1 for item in processed_images if item.used_subject_focused_crop)
    local_summary = {
        "used_borders_count": used_borders_count,
        "safe_crop_count": safe_crop_count,
        "subject_focused_crop_count": subject_crop_count,
        "rotated_on_pdf_count": sum(1 for item in processed_images if item.rotated_on_pdf),
        "analyzed_images_count": sum(1 for item in processed_images if item.analysis_report is not None),
    }
    strategy = _suggest_batch_strategy(processed_images, user_preferences)
    return BatchReport(
        total_images=len(processed_images),
        vertical_count=vertical_count,
        horizontal_count=horizontal_count,
        square_count=square_count,
        low_risk_count=low_risk_count,
        medium_risk_count=medium_risk_count,
        high_risk_count=high_risk_count,
        faces_total=faces_total,
        persons_total=persons_total,
        text_images_count=text_images_count,
        images=[_compact_image_report(item) for item in processed_images],
        user_preferences=dict(user_preferences),
        local_summary=local_summary,
        suggested_local_batch_strategy=strategy,
        warnings=warnings,
    )


def assert_no_image_data_in_ai_payload(payload: Any) -> None:
    if isinstance(payload, (bytes, bytearray, memoryview, Image.Image)):
        raise ValueError("Payload da IA contem dados de imagem ou binarios.")

    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in FORBIDDEN_AI_PAYLOAD_KEYS:
                raise ValueError(f"Payload da IA contem campo proibido: {key}")
            assert_no_image_data_in_ai_payload(value)
        return

    if isinstance(payload, (list, tuple)):
        for value in payload:
            assert_no_image_data_in_ai_payload(value)
        return

    if isinstance(payload, str):
        lowered = payload.lower()
        if "data:image" in lowered or "base64," in lowered:
            raise ValueError("Payload da IA contem imagem embutida.")
        if re.match(r"^[a-zA-Z]:[\\/]", payload) or payload.startswith("\\\\"):
            raise ValueError("Payload da IA contem caminho local absoluto.")


def build_batch_ai_payload(batch_report: BatchReport) -> dict[str, Any]:
    payload = {
        "total_images": batch_report.total_images,
        "user_preferences": batch_report.user_preferences,
        "summary": {
            "vertical_count": batch_report.vertical_count,
            "horizontal_count": batch_report.horizontal_count,
            "square_count": batch_report.square_count,
            "low_risk_count": batch_report.low_risk_count,
            "medium_risk_count": batch_report.medium_risk_count,
            "high_risk_count": batch_report.high_risk_count,
            "faces_total": batch_report.faces_total,
            "persons_total": batch_report.persons_total,
            "text_images_count": batch_report.text_images_count,
            "suggested_local_batch_strategy": batch_report.suggested_local_batch_strategy,
        },
        "images": batch_report.images,
        "warnings": batch_report.warnings,
    }
    assert_no_image_data_in_ai_payload(payload)
    return payload


def _final_strategy_for_item(item: ProcessedImage, user_preferences: dict[str, Any]) -> str:
    report = item.analysis_report
    max_safe_crop = float(user_preferences.get("max_safe_crop_percent") or DEFAULT_MAX_SAFE_CROP_PERCENT)
    priority = user_preferences.get("priority_mode")

    if _is_high_or_unsafe(item):
        return "contain_with_borders"
    if item.used_subject_focused_crop:
        return "subject_focused_crop"
    if (
        report
        and report.suggested_strategy == "subject_focused_crop"
        and report.recommended_crop_mode == "subject_focused_crop"
        and report.can_tighten_frame
    ):
        return "subject_focused_crop"
    if report and report.required_crop_percent > max_safe_crop:
        return "contain_with_borders"
    if report and report.edge_importance_max > MAX_EDGE_IMPORTANCE_FOR_SAFE_CROP and report.crop_amount_class != "tiny":
        return "contain_with_borders"
    if priority == PRIORITY_SAFE and _risk_level(item) != "low":
        return "contain_with_borders"
    if item.used_smart_crop and item.faces_detected:
        return "smart_face_crop"
    if item.used_safe_crop:
        return "safe_crop"
    if priority == PRIORITY_FILL_PHOTO and report and _risk_level(item) in {"low", "medium"}:
        if report.required_crop_percent <= max_safe_crop:
            return "safe_crop"
    if report and report.suggested_strategy in {"safe_crop", "smart_face_crop", "center_crop"}:
        return report.suggested_strategy
    if item.final_decision_strategy in {"safe_crop", "smart_face_crop", "center_crop"}:
        return item.final_decision_strategy
    return "contain_with_borders" if item.used_borders else "safe_crop"


def _decision_from_item(item: ProcessedImage, user_preferences: dict[str, Any], source: str) -> FinalImageDecision:
    strategy = _final_strategy_for_item(item, user_preferences)
    if strategy == "manual_review":
        strategy = "manual_review_fallback"
    return FinalImageDecision(
        image_name=item.output_name,
        final_strategy=strategy,
        crop_allowed=strategy in {"safe_crop", "subject_focused_crop", "smart_face_crop", "center_crop"},
        use_borders=strategy in {"contain_with_borders", "manual_review_fallback", "create_extra_page"},
        rotate_on_pdf=False,
        create_extra_page=False,
        pdf_page_number=None,
        pdf_slot_position=None,
        reason="Decisao local consolidada para o lote.",
        source=source,
        validation_notes=[],
    )


def _fit_strategy_for_decision(decision: FinalImageDecision, rotate_on_pdf: bool) -> str:
    if decision.create_extra_page:
        return "extra_page"
    if rotate_on_pdf:
        return "rotate_on_pdf"
    if decision.use_borders:
        return "contain_with_borders"
    if decision.final_strategy in {"safe_crop", "subject_focused_crop", "smart_face_crop", "center_crop"}:
        return "safe_crop"
    return "normal"


def _slot_for_item(
    item: ProcessedImage,
    decision: FinalImageDecision,
    position: str,
    slot_type: str,
    rotate_on_pdf: bool,
    reason: str,
) -> PlannedSlot:
    return PlannedSlot(
        position=position,
        slot_type=slot_type,
        image_name=item.output_name,
        rotate_on_pdf=rotate_on_pdf,
        fit_strategy=_fit_strategy_for_decision(decision, rotate_on_pdf),
        reason=reason,
    )


def _page_layout_type(slots: list[PlannedSlot]) -> str:
    if len(slots) == 1:
        return "single_photo"
    if len(slots) == 2 and all(slot.slot_type == ORIENTATION_HORIZONTAL for slot in slots):
        return "2_horizontal"
    if len(slots) == 3:
        return "3_photos_mixed"
    return "custom_safe"


def _make_page(page_number: int, slots: list[PlannedSlot], reason: str) -> PlannedPage:
    return PlannedPage(
        page_number=page_number,
        layout_type=_page_layout_type(slots),
        slots=slots,
        reason=reason,
        warnings=[],
    )


def _plan_best_fit_pages(
    processed_images: list[ProcessedImage],
    decisions: dict[str, FinalImageDecision],
    user_preferences: dict[str, Any],
) -> list[PlannedPage]:
    horizontals = [item for item in processed_images if _is_horizontal(item)]
    verticals = [item for item in processed_images if not _is_horizontal(item)]
    allow_rotate_vertical = user_preferences.get("priority_mode") != PRIORITY_SAFE
    pages: list[PlannedPage] = []

    while horizontals or verticals:
        page_number = len(pages) + 1
        slots: list[PlannedSlot] = []

        if horizontals and verticals:
            top = horizontals.pop(0)
            slots.append(
                _slot_for_item(top, decisions[top.output_name], "top", ORIENTATION_HORIZONTAL, False, "Foto horizontal no slot superior.")
            )
            for position in ("bottom_left", "bottom_right"):
                if verticals:
                    item = verticals.pop(0)
                    slots.append(
                        _slot_for_item(item, decisions[item.output_name], position, ORIENTATION_VERTICAL, False, "Foto vertical em slot vertical.")
                    )
            pages.append(_make_page(page_number, slots, "Melhor encaixe por orientacao."))
            continue

        if verticals:
            if len(verticals) >= 3 and allow_rotate_vertical:
                top = verticals.pop(0)
                slots.append(
                    _slot_for_item(top, decisions[top.output_name], "top", ORIENTATION_HORIZONTAL, True, "Vertical rotacionada apenas no PDF.")
                )
                for position in ("bottom_left", "bottom_right"):
                    item = verticals.pop(0)
                    slots.append(
                        _slot_for_item(item, decisions[item.output_name], position, ORIENTATION_VERTICAL, False, "Foto vertical em slot vertical.")
                    )
            elif len(verticals) >= 2:
                for position in ("bottom_left", "bottom_right"):
                    item = verticals.pop(0)
                    slots.append(
                        _slot_for_item(item, decisions[item.output_name], position, ORIENTATION_VERTICAL, False, "Pagina segura com fotos verticais.")
                    )
            else:
                item = verticals.pop(0)
                slots.append(
                    _slot_for_item(item, decisions[item.output_name], "center", ORIENTATION_VERTICAL, False, "Foto unica centralizada.")
                )
            pages.append(_make_page(page_number, slots, "Plano local para fotos verticais."))
            continue

        page_items = [horizontals.pop(0)]
        if horizontals:
            page_items.append(horizontals.pop(0))
        positions = ["top_1", "top_2"] if len(page_items) == 2 else ["center"]
        for item, position in zip(page_items, positions):
            slots.append(
                _slot_for_item(item, decisions[item.output_name], position, ORIENTATION_HORIZONTAL, False, "Foto horizontal em slot horizontal.")
            )
        pages.append(_make_page(page_number, slots, "Plano local para fotos horizontais."))

    return pages


def _plan_preserve_order_pages(
    processed_images: list[ProcessedImage],
    decisions: dict[str, FinalImageDecision],
    user_preferences: dict[str, Any],
) -> list[PlannedPage]:
    remaining = list(processed_images)
    pages: list[PlannedPage] = []
    allow_rotate_vertical = user_preferences.get("priority_mode") not in {PRIORITY_SAFE}

    while remaining:
        page_number = len(pages) + 1
        first = remaining.pop(0)
        slots: list[PlannedSlot] = []

        if _is_horizontal(first):
            slots.append(
                _slot_for_item(first, decisions[first.output_name], "top", ORIENTATION_HORIZONTAL, False, "Preservei a primeira foto da pagina.")
            )
            verticals = [item for item in remaining if not _is_horizontal(item)]
            for position in ("bottom_left", "bottom_right"):
                if not verticals:
                    break
                item = verticals.pop(0)
                remaining.remove(item)
                slots.append(
                    _slot_for_item(item, decisions[item.output_name], position, ORIENTATION_VERTICAL, False, "Completei com foto vertical segura.")
                )
            pages.append(_make_page(page_number, slots, "Preservei a ordem sempre que possivel."))
            continue

        if allow_rotate_vertical and len([item for item in remaining if not _is_horizontal(item)]) >= 2:
            slots.append(
                _slot_for_item(first, decisions[first.output_name], "top", ORIENTATION_HORIZONTAL, True, "Vertical rotacionada apenas no PDF para preservar ordem.")
            )
            for position in ("bottom_left", "bottom_right"):
                item = next((candidate for candidate in remaining if not _is_horizontal(candidate)), None)
                if item is None:
                    break
                remaining.remove(item)
                slots.append(
                    _slot_for_item(item, decisions[item.output_name], position, ORIENTATION_VERTICAL, False, "Foto vertical em slot vertical.")
                )
        else:
            slots.append(
                _slot_for_item(first, decisions[first.output_name], "center", ORIENTATION_VERTICAL, False, "Foto centralizada para evitar encaixe ruim.")
            )
        pages.append(_make_page(page_number, slots, "Plano local preservando upload."))

    return pages


def _plan_real_size_pages(
    processed_images: list[ProcessedImage],
    decisions: dict[str, FinalImageDecision],
) -> list[PlannedPage]:
    pages: list[PlannedPage] = []
    for start in range(0, len(processed_images), 2):
        page_number = len(pages) + 1
        group = processed_images[start : start + 2]
        positions = ["top_1", "top_2"] if len(group) == 2 else ["center"]
        slots = [
            _slot_for_item(
                item,
                decisions[item.output_name],
                position,
                ORIENTATION_HORIZONTAL if _is_horizontal(item) else ORIENTATION_VERTICAL,
                False,
                "Foto em tamanho fisico real no PDF.",
            )
            for item, position in zip(group, positions)
        ]
        pages.append(
            PlannedPage(
                page_number=page_number,
                layout_type="2_real_size" if len(slots) == 2 else "single_photo",
                slots=slots,
                reason="Layout 10x15 real com ate duas fotos por pagina.",
                warnings=[],
            )
        )
    return pages


def _plan_4up_real_pages(
    processed_images: list[ProcessedImage],
    decisions: dict[str, FinalImageDecision],
    preserve_order: bool,
) -> list[PlannedPage]:
    ordered_images = list(processed_images)
    if not preserve_order:
        ordered_images = [item for item in processed_images if not _is_horizontal(item)]
        ordered_images.extend(item for item in processed_images if _is_horizontal(item))

    pages: list[PlannedPage] = []
    positions = ("top_left", "top_right", "bottom_left", "bottom_right")
    for start in range(0, len(ordered_images), 4):
        page_number = len(pages) + 1
        group = ordered_images[start : start + 4]
        slots: list[PlannedSlot] = []
        for item, position in zip(group, positions):
            slot = _slot_for_item(
                item,
                decisions[item.output_name],
                position,
                ORIENTATION_VERTICAL,
                False,
                "Imagem em slot real 10x14,52.",
            )
            if _is_horizontal(item):
                slot = replace(
                    slot,
                    rotate_on_pdf=True,
                    fit_strategy="rotate_on_pdf",
                    reason="Foto horizontal rotacionada apenas no PDF para caber no slot 10x14,52.",
                )
            slots.append(slot)
        pages.append(
            PlannedPage(
                page_number=page_number,
                layout_type="4_real_images_a4",
                slots=slots,
                reason="Layout 4 imagens 10x14,52 por A4.",
                warnings=[],
            )
        )
    return pages


def _pop_matching_items(
    items: list[ProcessedImage],
    predicate,
    limit: int,
) -> tuple[list[ProcessedImage], bool]:
    selected: list[ProcessedImage] = []
    skipped_non_matching = False
    index = 0
    while index < len(items) and len(selected) < limit:
        if predicate(items[index]):
            selected.append(items.pop(index))
        else:
            skipped_non_matching = True
            index += 1
    return selected, skipped_non_matching


def _real_three_layout_type(slots: list[PlannedSlot]) -> str:
    return "3_real_photos_a4" if len(slots) == 3 else _page_layout_type(slots)


def _make_real_three_page(page_number: int, slots: list[PlannedSlot], reason: str) -> PlannedPage:
    return PlannedPage(
        page_number=page_number,
        layout_type=_real_three_layout_type(slots),
        slots=slots,
        reason=reason,
        warnings=[],
    )


def _plan_3_real_photos_best_fit(
    processed_images: list[ProcessedImage],
    decisions: dict[str, FinalImageDecision],
) -> list[PlannedPage]:
    horizontals = [item for item in processed_images if _is_horizontal(item)]
    verticals = [item for item in processed_images if not _is_horizontal(item)]
    pages: list[PlannedPage] = []

    while horizontals or verticals:
        page_number = len(pages) + 1
        slots: list[PlannedSlot] = []

        if horizontals and verticals:
            top = horizontals.pop(0)
            slots.append(
                _slot_for_item(top, decisions[top.output_name], "top", ORIENTATION_HORIZONTAL, False, "Foto horizontal no slot superior real 15x10.")
            )
            for position in ("bottom_left", "bottom_right"):
                if verticals:
                    item = verticals.pop(0)
                    slots.append(
                        _slot_for_item(item, decisions[item.output_name], position, ORIENTATION_VERTICAL, False, "Foto vertical em slot real 10x15.")
                    )
            pages.append(_make_real_three_page(page_number, slots, "Layout 3 fotos 10x15 reais por A4."))
            continue

        if verticals:
            if len(verticals) >= 3:
                top = verticals.pop(0)
                slots.append(
                    _slot_for_item(top, decisions[top.output_name], "top", ORIENTATION_HORIZONTAL, True, "Vertical rotacionada apenas no PDF para o slot real 15x10.")
                )
                for position in ("bottom_left", "bottom_right"):
                    item = verticals.pop(0)
                    slots.append(
                        _slot_for_item(item, decisions[item.output_name], position, ORIENTATION_VERTICAL, False, "Foto vertical em slot real 10x15.")
                    )
            elif len(verticals) == 2:
                for position in ("bottom_left", "bottom_right"):
                    item = verticals.pop(0)
                    slots.append(
                        _slot_for_item(item, decisions[item.output_name], position, ORIENTATION_VERTICAL, False, "Foto vertical em tamanho real.")
                    )
            else:
                item = verticals.pop(0)
                slots.append(
                    _slot_for_item(item, decisions[item.output_name], "center", ORIENTATION_VERTICAL, False, "Foto vertical unica em tamanho real.")
                )
            pages.append(_make_real_three_page(page_number, slots, "Plano local do layout 10x15 real."))
            continue

        item = horizontals.pop(0)
        slots.append(
            _slot_for_item(item, decisions[item.output_name], "center", ORIENTATION_HORIZONTAL, False, "Foto horizontal unica em tamanho real.")
        )
        pages.append(_make_real_three_page(page_number, slots, "Plano local do layout 10x15 real."))

    return pages


def _plan_3_real_photos_preserve_order(
    processed_images: list[ProcessedImage],
    decisions: dict[str, FinalImageDecision],
) -> list[PlannedPage]:
    remaining = list(processed_images)
    pages: list[PlannedPage] = []

    while remaining:
        page_number = len(pages) + 1
        first = remaining.pop(0)
        slots: list[PlannedSlot] = []
        remaining_verticals = [item for item in remaining if not _is_horizontal(item)]

        if _is_horizontal(first):
            if not remaining_verticals:
                slots.append(
                    _slot_for_item(first, decisions[first.output_name], "center", ORIENTATION_HORIZONTAL, False, "Foto horizontal unica em tamanho real.")
                )
                pages.append(_make_real_three_page(page_number, slots, "Preservei a ordem no layout 10x15 real."))
                continue

            slots.append(
                _slot_for_item(first, decisions[first.output_name], "top", ORIENTATION_HORIZONTAL, False, "Foto horizontal no slot superior real 15x10.")
            )
            bottoms, _ = _pop_matching_items(remaining, lambda item: not _is_horizontal(item), 2)
            for item, position in zip(bottoms, ("bottom_left", "bottom_right")):
                slots.append(
                    _slot_for_item(item, decisions[item.output_name], position, ORIENTATION_VERTICAL, False, "Foto vertical em slot real 10x15.")
                )
            pages.append(_make_real_three_page(page_number, slots, "Preservei a ordem sempre que possivel."))
            continue

        if len(remaining_verticals) >= 2:
            slots.append(
                _slot_for_item(first, decisions[first.output_name], "top", ORIENTATION_HORIZONTAL, True, "Vertical rotacionada apenas no PDF para o slot real 15x10.")
            )
            bottoms, _ = _pop_matching_items(remaining, lambda item: not _is_horizontal(item), 2)
            for item, position in zip(bottoms, ("bottom_left", "bottom_right")):
                slots.append(
                    _slot_for_item(item, decisions[item.output_name], position, ORIENTATION_VERTICAL, False, "Foto vertical em slot real 10x15.")
                )
        elif len(remaining_verticals) == 1:
            second, _ = _pop_matching_items(remaining, lambda item: not _is_horizontal(item), 1)
            pair = [first] + second
            for item, position in zip(pair, ("bottom_left", "bottom_right")):
                slots.append(
                    _slot_for_item(item, decisions[item.output_name], position, ORIENTATION_VERTICAL, False, "Foto vertical em tamanho real.")
                )
        else:
            slots.append(
                _slot_for_item(first, decisions[first.output_name], "center", ORIENTATION_VERTICAL, False, "Foto vertical unica em tamanho real.")
            )
        pages.append(_make_real_three_page(page_number, slots, "Preservei a ordem no layout 10x15 real."))

    return pages


def _plan_3_real_photos_pages(
    processed_images: list[ProcessedImage],
    decisions: dict[str, FinalImageDecision],
    preserve_order: bool,
) -> list[PlannedPage]:
    if preserve_order:
        return _plan_3_real_photos_preserve_order(processed_images, decisions)
    return _plan_3_real_photos_best_fit(processed_images, decisions)


def _attach_page_destinations(plan: BatchPlan) -> BatchPlan:
    decisions = {decision.image_name: decision for decision in plan.image_decisions}
    for page in plan.pages:
        for slot in page.slots:
            if not slot.image_name or slot.image_name not in decisions:
                continue
            decision = decisions[slot.image_name]
            decisions[slot.image_name] = replace(
                decision,
                rotate_on_pdf=slot.rotate_on_pdf,
                create_extra_page=decision.create_extra_page or slot.fit_strategy == "extra_page",
                pdf_page_number=page.page_number,
                pdf_slot_position=slot.position,
            )
    return replace(plan, image_decisions=list(decisions.values()))


def build_local_batch_plan(
    processed_images: list[ProcessedImage],
    batch_report: BatchReport,
    user_preferences: dict[str, Any],
) -> BatchPlan:
    decisions = {
        item.output_name: _decision_from_item(item, user_preferences, "local")
        for item in processed_images
    }
    preserve_order = (
        bool(user_preferences.get("preserve_upload_order"))
        or user_preferences.get("pdf_organization_mode") == PDF_ORGANIZE_UPLOAD_ORDER
    )
    if user_preferences.get("pdf_layout_mode") == PDF_LAYOUT_2_REAL:
        pages = _plan_real_size_pages(processed_images, decisions)
        strategy = BATCH_STRATEGY_PRESERVE_ORDER if preserve_order else BATCH_STRATEGY_BEST_FIT
    elif user_preferences.get("pdf_layout_mode") == PDF_LAYOUT_4_REAL_PHOTOS:
        pages = _plan_4up_real_pages(processed_images, decisions, preserve_order)
        strategy = BATCH_STRATEGY_PRESERVE_ORDER if preserve_order else BATCH_STRATEGY_BEST_FIT
    elif user_preferences.get("pdf_layout_mode") == PDF_LAYOUT_3_REAL_PHOTOS:
        pages = _plan_3_real_photos_pages(processed_images, decisions, preserve_order)
        strategy = BATCH_STRATEGY_PRESERVE_ORDER if preserve_order else batch_report.suggested_local_batch_strategy
    elif preserve_order:
        pages = _plan_preserve_order_pages(processed_images, decisions, user_preferences)
        strategy = BATCH_STRATEGY_PRESERVE_ORDER
    else:
        pages = _plan_best_fit_pages(processed_images, decisions, user_preferences)
        strategy = batch_report.suggested_local_batch_strategy or BATCH_STRATEGY_BEST_FIT

    plan = BatchPlan(
        source="local",
        strategy=strategy,
        pages=pages,
        image_decisions=list(decisions.values()),
        global_warnings=list(batch_report.warnings),
        explanation="Planejamento local criado a partir de orientacao, risco e preferencias do usuario.",
        confidence=1.0,
        validated=True,
        validation_notes=["Plano local usado como base segura."],
    )
    logger.info("Plano local do lote criado com %s pagina(s).", len(pages))
    return _attach_page_destinations(plan)
