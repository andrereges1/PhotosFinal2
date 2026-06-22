"""Cliente e parser da IA para planejamento do lote inteiro."""

from __future__ import annotations

from typing import Any

from src.ai_client import call_ai_batch_planner, extract_json_from_ai_response
from src.ai_prompts import build_batch_planning_prompt
from src.analysis_models import BatchPlan, BatchReport, FinalImageDecision, PlannedPage, PlannedSlot
from src.batch_planner import build_batch_ai_payload
from src.config import (
    BATCH_STRATEGY_BEST_FIT,
    DEFAULT_MAX_SAFE_CROP_PERCENT,
)


def _string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip() or default


def _bool(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _parse_slot(raw_slot: Any) -> PlannedSlot | None:
    if not isinstance(raw_slot, dict):
        return None
    return PlannedSlot(
        position=_string(raw_slot.get("position"), "center"),
        slot_type=_string(raw_slot.get("slot_type"), "vertical"),
        image_name=_string(raw_slot.get("image_name")) or None,
        rotate_on_pdf=_bool(raw_slot.get("rotate_on_pdf"), False),
        fit_strategy=_string(raw_slot.get("fit_strategy"), "normal"),
        reason=_string(raw_slot.get("reason"), "Slot sugerido pela IA."),
    )


def _parse_page(raw_page: Any) -> PlannedPage | None:
    if not isinstance(raw_page, dict):
        return None
    slots = [slot for slot in (_parse_slot(raw_slot) for raw_slot in _list(raw_page.get("slots"))) if slot]
    return PlannedPage(
        page_number=max(1, int(_float(raw_page.get("page_number"), 1))),
        layout_type=_string(raw_page.get("layout_type"), "custom_safe"),
        slots=slots,
        reason=_string(raw_page.get("reason"), "Pagina sugerida pela IA."),
        warnings=[_string(warning) for warning in _list(raw_page.get("warnings")) if _string(warning)],
    )


def _parse_decision(raw_decision: Any) -> FinalImageDecision | None:
    if not isinstance(raw_decision, dict):
        return None
    image_name = _string(raw_decision.get("image_name"))
    if not image_name:
        return None
    decision = _string(raw_decision.get("decision"), "contain_with_borders")
    use_borders = _bool(raw_decision.get("use_borders"), decision in {"contain_with_borders", "manual_review_fallback"})
    return FinalImageDecision(
        image_name=image_name,
        final_strategy=decision,
        crop_allowed=decision in {"safe_crop", "subject_focused_crop", "smart_face_crop", "center_crop"} and not use_borders,
        use_borders=use_borders,
        rotate_on_pdf=_bool(raw_decision.get("rotate_on_pdf"), False),
        create_extra_page=_bool(raw_decision.get("create_extra_page"), False),
        pdf_page_number=None,
        pdf_slot_position=None,
        reason=_string(raw_decision.get("reason"), "Decisao sugerida pela IA."),
        source="ai",
        validation_notes=[],
    )


def parse_batch_plan_response(response_text: str) -> BatchPlan | None:
    raw = extract_json_from_ai_response(response_text)
    if not isinstance(raw, dict):
        return None

    pages = [page for page in (_parse_page(raw_page) for raw_page in _list(raw.get("pages"))) if page]
    decisions = [
        decision
        for decision in (_parse_decision(raw_decision) for raw_decision in _list(raw.get("image_decisions")))
        if decision
    ]
    if not pages or not decisions:
        return None

    confidence = max(0.0, min(1.0, _float(raw.get("confidence"), 0.0)))
    return BatchPlan(
        source="ai",
        strategy=_string(raw.get("strategy"), BATCH_STRATEGY_BEST_FIT),
        pages=pages,
        image_decisions=decisions,
        global_warnings=[_string(warning) for warning in _list(raw.get("global_warnings")) if _string(warning)],
        explanation=_string(raw.get("explanation"), "Plano sugerido pela IA de lote."),
        confidence=confidence,
        validated=False,
        validation_notes=[],
    )


def request_ai_batch_plan(batch_report: BatchReport) -> tuple[BatchPlan | None, dict[str, Any] | None, str | None, str | None, str | None]:
    payload = build_batch_ai_payload(batch_report)
    prompt = build_batch_planning_prompt(payload)
    response = call_ai_batch_planner(prompt)
    if response.response_text is None:
        return None, payload, prompt, None, response.error or "IA de lote nao respondeu"

    plan = parse_batch_plan_response(response.response_text)
    if plan is None:
        return None, payload, prompt, response.response_text, "Resposta da IA de lote nao era JSON valido"

    if plan.confidence <= 0:
        plan.confidence = DEFAULT_MAX_SAFE_CROP_PERCENT / 100
    return plan, payload, prompt, response.response_text, None
