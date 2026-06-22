"""Motor final de decisao do lote inteiro."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from src.analysis_models import BatchPlan, BatchReport
from src.batch_ai_planner import request_ai_batch_plan
from src.batch_planner import build_batch_report, build_local_batch_plan
from src.layout_validator import validate_batch_plan
from src.utils import ProcessedImage

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FinalProcessingPlanResult:
    batch_report: BatchReport
    local_plan: BatchPlan
    final_plan: BatchPlan
    ai_payload: dict[str, Any] | None = None
    ai_prompt: str | None = None
    ai_raw_response_text: str | None = None
    ai_plan: BatchPlan | None = None
    ai_error: str | None = None
    validation_notes: list[str] | None = None
    used_ai: bool = False
    fallback_used: bool = False
    corrected: bool = False

    def technical_payload(self) -> dict[str, Any]:
        return {
            "batch_report": self.batch_report.to_dict(),
            "ai_payload": self.ai_payload,
            "ai_prompt": self.ai_prompt,
            "ai_raw_response_text": self.ai_raw_response_text,
            "ai_plan": self.ai_plan.to_dict() if self.ai_plan else None,
            "validation_notes": self.validation_notes or [],
            "final_plan": self.final_plan.to_dict(),
            "ai_error": self.ai_error,
            "used_ai": self.used_ai,
            "fallback_used": self.fallback_used,
            "corrected": self.corrected,
        }


def _apply_plan_metadata(processed_images: list[ProcessedImage], plan: BatchPlan) -> None:
    items = {item.output_name: item for item in processed_images}
    for item in processed_images:
        item.batch_plan_source = plan.source
        item.batch_validation_notes = list(plan.validation_notes)

    for decision in plan.image_decisions:
        item = items.get(decision.image_name)
        if item is None:
            continue
        item.batch_final_strategy = decision.final_strategy
        item.batch_pdf_slot_position = decision.pdf_slot_position
        item.ai_rotate_on_pdf_requested = bool(item.ai_rotate_on_pdf_requested or decision.rotate_on_pdf)
        item.ai_create_extra_page_requested = bool(item.ai_create_extra_page_requested or decision.create_extra_page)


def build_final_processing_plan(
    processed_images: list[ProcessedImage],
    batch_report: BatchReport | None,
    user_preferences: dict[str, Any],
    use_ai_batch_planning: bool,
) -> FinalProcessingPlanResult:
    batch_report = batch_report or build_batch_report(processed_images, user_preferences)
    local_plan = build_local_batch_plan(processed_images, batch_report, user_preferences)

    if not use_ai_batch_planning:
        _apply_plan_metadata(processed_images, local_plan)
        logger.info("IA de lote desativada; usando plano local.")
        return FinalProcessingPlanResult(
            batch_report=batch_report,
            local_plan=local_plan,
            final_plan=local_plan,
            validation_notes=local_plan.validation_notes,
            used_ai=False,
            fallback_used=False,
            corrected=False,
        )

    try:
        ai_plan, ai_payload, ai_prompt, ai_raw_response_text, ai_error = request_ai_batch_plan(batch_report)
    except Exception:
        logger.info("Falha inesperada na IA de lote; usando plano local.")
        ai_plan = None
        ai_payload = None
        ai_prompt = None
        ai_raw_response_text = None
        ai_error = "IA de lote indisponivel"

    validation = validate_batch_plan(
        ai_plan,
        batch_report,
        processed_images,
        user_preferences,
        local_plan=local_plan,
    )
    final_plan = validation.plan
    _apply_plan_metadata(processed_images, final_plan)

    if validation.fallback_used:
        logger.info("Plano local usado como fallback da IA de lote.")
    else:
        logger.info("Plano da IA de lote validado com %s pagina(s).", len(final_plan.pages))

    return FinalProcessingPlanResult(
        batch_report=batch_report,
        local_plan=local_plan,
        final_plan=final_plan,
        ai_payload=ai_payload,
        ai_prompt=ai_prompt,
        ai_raw_response_text=ai_raw_response_text,
        ai_plan=ai_plan,
        ai_error=ai_error,
        validation_notes=validation.notes,
        used_ai=ai_plan is not None and not validation.fallback_used,
        fallback_used=validation.fallback_used,
        corrected=validation.corrected,
    )
