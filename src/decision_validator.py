"""Validacao local rigida das sugestoes da IA."""

from __future__ import annotations

from dataclasses import replace

from src.analysis_models import AIDecision, ImageAnalysisReport
from src.config import (
    AI_ALLOWED_DECISIONS,
    AI_FALLBACK_DECISION,
    DEFAULT_MAX_SAFE_CROP_PERCENT,
    HIGH_RISK_CROP_PERCENT,
    MAX_EDGE_IMPORTANCE_FOR_SAFE_CROP,
)

CROP_DECISIONS = {"safe_crop", "subject_focused_crop", "smart_face_crop", "center_crop"}
CONSERVATIVE_DECISIONS = {"contain_with_borders", "create_extra_page", "manual_review"}


def _with_notes(ai_decision: AIDecision, notes: list[str], validated: bool) -> AIDecision:
    warnings = list(ai_decision.warnings)
    if not validated and notes:
        warnings.append(notes[-1])
    return replace(
        ai_decision,
        validated=validated,
        validation_notes=notes,
        warnings=list(dict.fromkeys(warnings)),
    )


def _force_borders(ai_decision: AIDecision, notes: list[str]) -> AIDecision:
    reason = notes[-1] if notes else "Usei bordas por seguranca."
    return replace(
        ai_decision,
        decision=AI_FALLBACK_DECISION,
        reason=reason,
        risk_level="high",
        use_borders=True,
        allow_crop=False,
        rotate_on_pdf=False,
        create_extra_page=False,
        validated=False,
        validation_notes=notes,
        warnings=list(dict.fromkeys(list(ai_decision.warnings) + [reason])),
    )


def _is_crop_decision(decision: str) -> bool:
    return decision in CROP_DECISIONS


def validate_ai_decision(
    ai_decision: AIDecision,
    image_report: ImageAnalysisReport,
    max_safe_crop_percent: float = DEFAULT_MAX_SAFE_CROP_PERCENT,
) -> AIDecision:
    notes: list[str] = []
    decision = ai_decision.decision

    if decision not in AI_ALLOWED_DECISIONS:
        notes.append("A IA sugeriu uma decisao desconhecida. Usei a decisao local.")
        return replace(
            ai_decision,
            decision=image_report.suggested_strategy if image_report.suggested_strategy in AI_ALLOWED_DECISIONS else AI_FALLBACK_DECISION,
            validated=False,
            validation_notes=notes,
        )

    max_crop = ai_decision.max_crop_percent
    if max_crop is not None and max_crop > max_safe_crop_percent:
        notes.append("Limitei o corte maximo ao valor permitido localmente.")
        ai_decision = replace(ai_decision, max_crop_percent=max_safe_crop_percent)

    if _is_crop_decision(decision):
        if image_report.faces_detected > 0 and not image_report.faces_safe_for_crop:
            notes.append("A IA sugeriu corte, mas a analise local detectou rosto em risco. Usei bordas.")
            return _force_borders(ai_decision, notes)

        if image_report.faces_near_edges:
            notes.append("A IA sugeriu corte, mas ha rosto perto da borda. Usei bordas.")
            return _force_borders(ai_decision, notes)

        if image_report.persons_detected > 0 and not image_report.persons_safe_for_crop:
            notes.append("A IA sugeriu corte, mas a analise local detectou pessoa em risco. Usei bordas.")
            return _force_borders(ai_decision, notes)

        if image_report.persons_near_edges:
            notes.append("A IA sugeriu corte, mas ha pessoa perto da borda. Usei bordas.")
            return _force_borders(ai_decision, notes)

        if image_report.text_detected and image_report.text_near_edges:
            notes.append("A IA sugeriu corte, mas a analise local detectou texto perto da borda. Usei bordas.")
            return _force_borders(ai_decision, notes)

        if image_report.required_crop_percent > min(max_safe_crop_percent, HIGH_RISK_CROP_PERCENT):
            notes.append("A IA sugeriu corte maior que o limite seguro. Usei bordas.")
            return _force_borders(ai_decision, notes)

        if image_report.edge_importance_max > MAX_EDGE_IMPORTANCE_FOR_SAFE_CROP and image_report.crop_amount_class != "tiny":
            notes.append("A IA sugeriu corte, mas as bordas parecem importantes. Usei bordas.")
            return _force_borders(ai_decision, notes)

        if image_report.risk_level == "high":
            notes.append("A IA sugeriu corte, mas o risco local e alto. Usei bordas.")
            return _force_borders(ai_decision, notes)

    if decision in CONSERVATIVE_DECISIONS:
        notes.append("A sugestao da IA e conservadora e foi aceita.")
        return _with_notes(ai_decision, notes, validated=True)

    if decision == "rotate_on_pdf":
        notes.append("A rotacao no PDF sera validada pelo gerador de PDF.")
        return _with_notes(ai_decision, notes, validated=True)

    notes.append("A sugestao da IA passou pelas regras locais.")
    return _with_notes(ai_decision, notes, validated=True)
