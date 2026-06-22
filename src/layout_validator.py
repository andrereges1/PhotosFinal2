"""Validador local do plano global sugerido pela IA."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from src.analysis_models import BatchPlan, BatchReport, FinalImageDecision, PlannedPage, PlannedSlot
from src.batch_planner import build_local_batch_plan
from src.config import (
    BATCH_ALLOWED_FINAL_STRATEGIES,
    BATCH_ALLOWED_FIT_STRATEGIES,
    BATCH_ALLOWED_LAYOUT_TYPES,
    BATCH_ALLOWED_SLOT_POSITIONS,
    BATCH_ALLOWED_STRATEGIES,
    BATCH_STRATEGY_BEST_FIT,
    DEFAULT_MAX_SAFE_CROP_PERCENT,
    MAX_EDGE_IMPORTANCE_FOR_SAFE_CROP,
    ORIENTATION_HORIZONTAL,
    ORIENTATION_VERTICAL,
)
from src.utils import ProcessedImage


@dataclass(slots=True)
class BatchPlanValidationResult:
    plan: BatchPlan
    notes: list[str]
    accepted: bool
    corrected: bool
    fallback_used: bool


def _fallback_result(local_plan: BatchPlan, notes: list[str]) -> BatchPlanValidationResult:
    fallback = replace(
        local_plan,
        source="fallback",
        validated=True,
        validation_notes=list(dict.fromkeys(list(local_plan.validation_notes) + notes)),
    )
    return BatchPlanValidationResult(
        plan=fallback,
        notes=fallback.validation_notes,
        accepted=False,
        corrected=False,
        fallback_used=True,
    )


def _is_horizontal(item: ProcessedImage) -> bool:
    if item.final_orientation:
        return item.final_orientation == ORIENTATION_HORIZONTAL
    return item.image.width >= item.image.height


def _report_lookup(batch_report: BatchReport) -> dict[str, dict[str, Any]]:
    return {
        str(image.get("image_name")): image
        for image in batch_report.images
        if image.get("image_name")
    }


def _must_use_borders(
    item: ProcessedImage,
    batch_image: dict[str, Any],
    user_preferences: dict[str, Any],
    final_strategy: str,
) -> str | None:
    report = item.analysis_report
    max_safe_crop = float(user_preferences.get("max_safe_crop_percent") or DEFAULT_MAX_SAFE_CROP_PERCENT)
    locally_validated_subject_crop = (
        final_strategy == "subject_focused_crop"
        and item.used_subject_focused_crop
        and report is not None
        and report.recommended_crop_mode == "subject_focused_crop"
        and report.can_tighten_frame
    )
    if report:
        if report.risk_level == "high":
            return "Risco local alto; usei bordas."
        if report.faces_detected > 0 and (report.faces_near_edges or not report.faces_safe_for_crop):
            return "Rosto em risco; usei bordas."
        if report.persons_detected > 0 and (report.persons_near_edges or not report.persons_safe_for_crop):
            return "Pessoa em risco; usei bordas."
        if report.text_detected and report.text_near_edges:
            return "Texto perto da borda; usei bordas."
        if report.required_crop_percent > max_safe_crop and not locally_validated_subject_crop:
            return "Corte maior que o limite local; usei bordas."
        if (
            report.edge_importance_max > MAX_EDGE_IMPORTANCE_FOR_SAFE_CROP
            and report.crop_amount_class != "tiny"
            and not locally_validated_subject_crop
        ):
            return "Bordas importantes; usei bordas."
    if batch_image.get("risk_level") == "high":
        return "Risco do lote alto; usei bordas."
    return None


def _validate_decisions(
    plan: BatchPlan,
    batch_report: BatchReport,
    processed_images: list[ProcessedImage],
    user_preferences: dict[str, Any],
) -> tuple[list[FinalImageDecision], list[str], bool, int]:
    notes: list[str] = []
    corrected = False
    structural_errors = 0
    item_by_name = {item.output_name: item for item in processed_images}
    batch_by_name = _report_lookup(batch_report)
    seen_decisions: set[str] = set()
    decisions: list[FinalImageDecision] = []

    for decision in plan.image_decisions:
        if decision.image_name not in item_by_name:
            notes.append(f"A IA citou foto inexistente: {decision.image_name}.")
            structural_errors += 1
            continue
        if decision.image_name in seen_decisions:
            notes.append(f"A IA duplicou decisao para {decision.image_name}.")
            structural_errors += 1
            continue
        seen_decisions.add(decision.image_name)

        final_strategy = decision.final_strategy
        if final_strategy not in BATCH_ALLOWED_FINAL_STRATEGIES:
            notes.append(f"Decisao desconhecida para {decision.image_name}.")
            structural_errors += 1
            continue

        item = item_by_name[decision.image_name]
        must_use_borders_reason = _must_use_borders(
            item,
            batch_by_name.get(decision.image_name, {}),
            user_preferences,
            final_strategy,
        )
        if final_strategy in {"safe_crop", "subject_focused_crop", "smart_face_crop", "center_crop"} and must_use_borders_reason:
            decision = replace(
                decision,
                final_strategy="contain_with_borders",
                crop_allowed=False,
                use_borders=True,
                reason=must_use_borders_reason,
                validation_notes=list(dict.fromkeys(decision.validation_notes + [must_use_borders_reason])),
            )
            notes.append(f"Troquei {decision.image_name} para bordas por seguranca.")
            corrected = True

        if decision.final_strategy == "manual_review":
            decision = replace(
                decision,
                final_strategy="manual_review_fallback",
                crop_allowed=False,
                use_borders=True,
                validation_notes=list(dict.fromkeys(decision.validation_notes + ["Manual review vira fallback com bordas."])),
            )
            corrected = True

        decisions.append(decision)

    missing_decisions = set(item_by_name) - seen_decisions
    if missing_decisions:
        notes.append("A IA esqueceu decisao de uma ou mais fotos.")
        structural_errors += len(missing_decisions)

    return decisions, notes, corrected, structural_errors


def _validate_pages(
    plan: BatchPlan,
    processed_images: list[ProcessedImage],
) -> tuple[list[PlannedPage], list[str], bool, int]:
    notes: list[str] = []
    corrected = False
    structural_errors = 0
    item_by_name = {item.output_name: item for item in processed_images}
    planned_names: list[str] = []
    pages: list[PlannedPage] = []
    seen_page_numbers: set[int] = set()

    for page in plan.pages:
        if page.page_number < 1:
            notes.append("A IA sugeriu numero de pagina invalido.")
            structural_errors += 1
            continue
        if page.page_number in seen_page_numbers:
            notes.append(f"A IA duplicou o numero da pagina {page.page_number}.")
            structural_errors += 1
            continue
        seen_page_numbers.add(page.page_number)

        if page.layout_type not in BATCH_ALLOWED_LAYOUT_TYPES:
            notes.append(f"Layout desconhecido na pagina {page.page_number}.")
            structural_errors += 1
            continue
        is_4up_real_layout = page.layout_type == "4_real_images_a4"
        slots: list[PlannedSlot] = []
        for slot in page.slots:
            if slot.position not in BATCH_ALLOWED_SLOT_POSITIONS:
                notes.append(f"Slot desconhecido na pagina {page.page_number}: {slot.position}.")
                structural_errors += 1
                continue
            if slot.slot_type not in {ORIENTATION_HORIZONTAL, ORIENTATION_VERTICAL}:
                notes.append(f"Tipo de slot desconhecido na pagina {page.page_number}.")
                structural_errors += 1
                continue
            if slot.fit_strategy not in BATCH_ALLOWED_FIT_STRATEGIES:
                notes.append(f"Estrategia de encaixe desconhecida em {slot.image_name}.")
                structural_errors += 1
                continue
            if not slot.image_name or slot.image_name not in item_by_name:
                notes.append(f"A IA citou foto inexistente no PDF: {slot.image_name}.")
                structural_errors += 1
                continue
            item = item_by_name[slot.image_name]
            if _is_horizontal(item) and slot.slot_type == ORIENTATION_VERTICAL and not is_4up_real_layout:
                notes.append(f"Foto horizontal em slot vertical ruim: {slot.image_name}.")
                structural_errors += 1
                continue
            if _is_horizontal(item) and slot.slot_type == ORIENTATION_VERTICAL and is_4up_real_layout:
                slot = replace(
                    slot,
                    rotate_on_pdf=True,
                    fit_strategy="rotate_on_pdf",
                    reason=(slot.reason or "Corrigido localmente.") + " Foto horizontal rotacionada no PDF para o layout 10x14,52.",
                )
                corrected = True
            if not _is_horizontal(item) and slot.slot_type == ORIENTATION_HORIZONTAL and not slot.rotate_on_pdf:
                slot = replace(
                    slot,
                    rotate_on_pdf=True,
                    fit_strategy="rotate_on_pdf",
                    reason=(slot.reason or "Corrigido localmente.") + " Rotacao no PDF exigida pelo validador.",
                )
                notes.append(f"Ativei rotacao no PDF para {slot.image_name}.")
                corrected = True
            planned_names.append(slot.image_name)
            slots.append(slot)

        max_slots = 4 if is_4up_real_layout else 3
        if len(slots) > max_slots:
            notes.append(f"Pagina {page.page_number} tem fotos demais para o layout escolhido.")
            structural_errors += 1
            continue
        if slots:
            pages.append(replace(page, slots=slots))

    duplicate_names = {name for name in planned_names if planned_names.count(name) > 1}
    if duplicate_names:
        notes.append("A IA colocou uma foto em mais de uma pagina.")
        structural_errors += len(duplicate_names)

    missing_names = set(item_by_name) - set(planned_names)
    if missing_names:
        notes.append("A IA deixou uma ou mais fotos sem destino no PDF.")
        structural_errors += len(missing_names)

    return pages, notes, corrected, structural_errors


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


def validate_batch_plan(
    ai_plan: BatchPlan | None,
    batch_report: BatchReport,
    processed_images: list[ProcessedImage],
    user_preferences: dict[str, Any],
    local_plan: BatchPlan | None = None,
) -> BatchPlanValidationResult:
    local_plan = local_plan or build_local_batch_plan(processed_images, batch_report, user_preferences)
    if ai_plan is None:
        return _fallback_result(local_plan, ["IA de lote indisponivel; usei plano local."])

    notes: list[str] = []
    corrected = False
    structural_errors = 0

    if ai_plan.strategy not in BATCH_ALLOWED_STRATEGIES:
        notes.append("A IA sugeriu estrategia global desconhecida.")
        ai_plan = replace(ai_plan, strategy=BATCH_STRATEGY_BEST_FIT)
        corrected = True

    decisions, decision_notes, decisions_corrected, decision_errors = _validate_decisions(
        ai_plan,
        batch_report,
        processed_images,
        user_preferences,
    )
    pages, page_notes, pages_corrected, page_errors = _validate_pages(ai_plan, processed_images)
    notes.extend(decision_notes)
    notes.extend(page_notes)
    corrected = corrected or decisions_corrected or pages_corrected
    structural_errors += decision_errors + page_errors

    if structural_errors >= 1:
        return _fallback_result(local_plan, notes + ["Plano da IA descartado por problema estrutural."])

    validated = BatchPlan(
        source="ai_validated",
        strategy=ai_plan.strategy,
        pages=pages,
        image_decisions=decisions,
        global_warnings=list(dict.fromkeys(ai_plan.global_warnings)),
        explanation=ai_plan.explanation or "Plano da IA validado localmente.",
        confidence=max(0.0, min(1.0, ai_plan.confidence)),
        validated=True,
        validation_notes=list(dict.fromkeys(notes + ["Plano da IA validado pelas regras locais."])),
    )
    validated = _attach_page_destinations(validated)
    return BatchPlanValidationResult(
        plan=validated,
        notes=validated.validation_notes,
        accepted=True,
        corrected=corrected,
        fallback_used=False,
    )
