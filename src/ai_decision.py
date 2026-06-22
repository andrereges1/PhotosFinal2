"""Orquestracao da decisao assistida por IA de texto."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from src.ai_client import call_ai_decision_model, load_ai_config, parse_ai_json_response
from src.ai_prompts import build_ai_decision_prompt
from src.analysis_models import AIDecision, ImageAnalysisReport
from src.config import AI_ALLOWED_DECISIONS, AI_FALLBACK_DECISION, DEFAULT_MAX_SAFE_CROP_PERCENT
from src.decision_validator import validate_ai_decision

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AIDecisionResult:
    decision: AIDecision | None
    report_payload: dict[str, Any] | None = None
    prompt: str | None = None
    raw_response_text: str | None = None
    raw_response: dict[str, Any] | None = None
    error: str | None = None


def _round(value: float | None, digits: int = 4) -> float:
    if value is None:
        return 0.0
    return round(float(value), digits)


def build_ai_report_payload(image_report: ImageAnalysisReport) -> dict[str, Any]:
    """Gera o unico conteudo que pode ser enviado para a IA: JSON tecnico."""
    return {
        "image_name": image_report.image_name,
        "orientation": image_report.orientation,
        "original_size": {
            "width": image_report.width,
            "height": image_report.height,
        },
        "target_format": image_report.target_format,
        "target_aspect_ratio": _round(image_report.target_aspect_ratio),
        "required_crop": {
            "axis": image_report.required_crop_axis,
            "percent": _round(image_report.required_crop_percent, 2),
            "class": image_report.crop_amount_class,
        },
        "faces": {
            "count": image_report.faces_detected,
            "near_edges": image_report.faces_near_edges,
            "safe_for_crop": image_report.faces_safe_for_crop,
            "multiple_faces": image_report.faces_detected > 1,
        },
        "persons": {
            "count": image_report.persons_detected,
            "near_edges": image_report.persons_near_edges,
            "safe_for_crop": image_report.persons_safe_for_crop,
        },
        "text": {
            "detected": image_report.text_detected,
            "near_edges": image_report.text_near_edges,
        },
        "edges": {
            "left_importance": _round(image_report.edge_importance_left),
            "right_importance": _round(image_report.edge_importance_right),
            "top_importance": _round(image_report.edge_importance_top),
            "bottom_importance": _round(image_report.edge_importance_bottom),
            "max_importance": _round(image_report.edge_importance_max),
        },
        "scores": {
            "visual_complexity": _round(image_report.visual_complexity_score),
            "center_importance": _round(image_report.center_importance_score),
            "border_importance": _round(image_report.border_importance_score),
        },
        "local_suggestion": {
            "strategy": image_report.suggested_strategy,
            "risk_level": image_report.risk_level,
            "reasons": image_report.reasons[:5],
        },
        "subject_focus": {
            "primary_subject_type": image_report.primary_subject_type,
            "primary_subject_box_summary": {
                "present": image_report.primary_subject_box is not None,
                "width_percent": _round(
                    (image_report.primary_subject_box[2] / max(1, image_report.width) * 100)
                    if image_report.primary_subject_box
                    else 0.0,
                    2,
                ),
                "height_percent": _round(
                    (image_report.primary_subject_box[3] / max(1, image_report.height) * 100)
                    if image_report.primary_subject_box
                    else 0.0,
                    2,
                ),
            },
            "subject_focus_score": _round(image_report.subject_focus_score),
            "background_waste_score": _round(image_report.background_waste_score),
            "can_tighten_frame": image_report.can_tighten_frame,
            "recommended_crop_mode": image_report.recommended_crop_mode,
        },
    }


def _bool_from_raw(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _float_from_raw(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float_from_raw(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _list_from_raw(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _decision_from_raw(image_name: str, raw_response: dict[str, Any]) -> AIDecision:
    decision = str(raw_response.get("decision", AI_FALLBACK_DECISION)).strip()
    if decision not in AI_ALLOWED_DECISIONS:
        decision = AI_FALLBACK_DECISION
    use_borders_default = decision in {"contain_with_borders", "manual_review", "create_extra_page"}
    allow_crop_default = decision in {"safe_crop", "subject_focused_crop", "smart_face_crop", "center_crop"}
    return AIDecision(
        image_name=image_name,
        decision=decision,
        confidence=max(0.0, min(1.0, _float_from_raw(raw_response.get("confidence"), 0.0))),
        reason=str(raw_response.get("reason", "")).strip() or "A IA nao informou motivo.",
        risk_level=str(raw_response.get("risk_level", "medium")).strip() or "medium",
        use_borders=_bool_from_raw(raw_response.get("use_borders"), use_borders_default),
        allow_crop=_bool_from_raw(raw_response.get("allow_crop"), allow_crop_default),
        max_crop_percent=_optional_float_from_raw(raw_response.get("max_crop_percent")),
        protect_faces=_bool_from_raw(raw_response.get("protect_faces"), True),
        protect_people=_bool_from_raw(raw_response.get("protect_people"), True),
        protect_text=_bool_from_raw(raw_response.get("protect_text"), True),
        rotate_on_pdf=_bool_from_raw(raw_response.get("rotate_on_pdf"), decision == "rotate_on_pdf"),
        create_extra_page=_bool_from_raw(raw_response.get("create_extra_page"), decision == "create_extra_page"),
        warnings=_list_from_raw(raw_response.get("warnings")),
        raw_response=raw_response,
        validated=False,
        validation_notes=[],
    )


def get_ai_decision_for_report(
    image_report: ImageAnalysisReport,
    max_safe_crop_percent: float = DEFAULT_MAX_SAFE_CROP_PERCENT,
) -> AIDecisionResult:
    report_payload = build_ai_report_payload(image_report)
    prompt = build_ai_decision_prompt(report_payload)
    config = load_ai_config()

    response = call_ai_decision_model(prompt, config)
    if response.response_text is None:
        return AIDecisionResult(
            decision=None,
            report_payload=report_payload,
            prompt=prompt,
            raw_response=response.raw_response,
            error=response.error or "IA nao respondeu",
        )

    parsed = parse_ai_json_response(response.response_text)
    if parsed is None:
        return AIDecisionResult(
            decision=None,
            report_payload=report_payload,
            prompt=prompt,
            raw_response_text=response.response_text,
            raw_response=response.raw_response,
            error="Resposta da IA nao era JSON valido",
        )

    ai_decision = _decision_from_raw(image_report.image_name, parsed)
    validated = validate_ai_decision(ai_decision, image_report, max_safe_crop_percent)
    logger.info("Decisao IA=%s final_validada=%s", ai_decision.decision, validated.decision)
    return AIDecisionResult(
        decision=validated,
        report_payload=report_payload,
        prompt=prompt,
        raw_response_text=response.response_text,
        raw_response=response.raw_response,
    )
