from __future__ import annotations

import logging

import streamlit as st

from src.config import (
    AI_DECISION_ENABLED_DEFAULT,
    FORMAT_10X15_VERTICAL,
    FORMAT_15X10_HORIZONTAL,
    FORMAT_AUTO,
    FRAME_PRIORITY_BALANCED,
    FRAME_PRIORITY_KEEP_SCENE,
    FRAME_PRIORITY_PEOPLE,
    ORIENTATION_HORIZONTAL,
    ORIENTATION_SQUARE,
    ORIENTATION_VERTICAL,
    PAGE_MARGIN_CM,
    PDF_4UP_HORIZONTAL_GAP_CM,
    PDF_4UP_VERTICAL_GAP_CM,
    PDF_EDGE_MARGIN_CM,
    PDF_ORGANIZE_AUTO,
    PDF_ORGANIZE_UPLOAD_ORDER,
    PDF_PHOTO_GAP_CM,
    PDF_LAYOUT_2_REAL,
    PDF_LAYOUT_3_PER_A4,
    PDF_LAYOUT_3_REAL_PHOTOS,
    PDF_LAYOUT_4_REAL_PHOTOS,
    PDF_LAYOUT_LABELS,
    PRIORITY_BALANCED,
    PRIORITY_FILL_PHOTO,
    PRIORITY_PAPER_SAVING,
    PRIORITY_SAFE,
    RESIZE_CONTAIN,
    RESIZE_COVER,
    RESIZE_SAFE_CROP,
    RESIZE_SMART,
)
from src.final_decision_engine import FinalProcessingPlanResult, build_final_processing_plan
from src.image_processing import generate_preview_image, open_image_safely, process_image
from src.pdf_generator import PdfGenerationResult, generate_pdf_with_summary
from src.utils import PdfOptions, ProcessingOptions, ensure_output_dirs
from src.zip_generator import generate_zip_bytes


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ensure_output_dirs()


FORMAT_OPTIONS = {
    "Automático": FORMAT_AUTO,
    "10x15 vertical": FORMAT_10X15_VERTICAL,
    "15x10 horizontal": FORMAT_15X10_HORIZONTAL,
}

RESIZE_OPTIONS = {
    "Automático inteligente": RESIZE_SMART,
    "Preencher com corte seguro": RESIZE_SAFE_CROP,
    "Manter foto inteira com bordas": RESIZE_CONTAIN,
    "Preencher e cortar o mínimo necessário": RESIZE_COVER,
}

PDF_OPTIONS = {
    "3 fotos 10x15 reais por A4": PDF_LAYOUT_3_REAL_PHOTOS,
    "4 imagens 10x14,52 por A4": PDF_LAYOUT_4_REAL_PHOTOS,
    "3 fotos em proporção 10x15": PDF_LAYOUT_3_PER_A4,
    "2 fotos 10x15 reais": PDF_LAYOUT_2_REAL,
}

GENERATE_OPTIONS = {
    "Gerar os dois": "both",
    "Ajustar fotos individuais": "images",
    "Gerar PDF A4": "pdf",
}

PDF_ORGANIZE_OPTIONS = {
    "Organizar automaticamente pelo melhor encaixe": PDF_ORGANIZE_AUTO,
    "Manter a ordem em que enviei": PDF_ORGANIZE_UPLOAD_ORDER,
}

PRIORITY_OPTIONS = {
    "Equilibrado": PRIORITY_BALANCED,
    "Mais seguro": PRIORITY_SAFE,
    "Economizar papel": PRIORITY_PAPER_SAVING,
    "Preencher mais a foto": PRIORITY_FILL_PHOTO,
}

FRAME_PRIORITY_OPTIONS = {
    "Preservar mais cenario": FRAME_PRIORITY_KEEP_SCENE,
    "Equilibrado": FRAME_PRIORITY_BALANCED,
    "Foco nas pessoas": FRAME_PRIORITY_PEOPLE,
}


def _reset_results_if_needed(upload_signature: tuple[tuple[str, int | None], ...]) -> None:
    current_signature = st.session_state.get("last_upload_signature")
    if current_signature != upload_signature:
        st.session_state.pop("results", None)
        st.session_state["last_upload_signature"] = upload_signature


def _show_pdf_summary(pdf_result: PdfGenerationResult | None) -> None:
    if not pdf_result:
        return

    st.markdown("### Resumo do PDF")
    for warning in pdf_result.warnings:
        st.info(warning)

    rotated_count = sum(1 for placement in pdf_result.placements if placement.rotated_on_pdf)
    if rotated_count:
        st.info(f"{rotated_count} foto(s) foram rotacionadas apenas no PDF para aproveitar melhor a folha.")

    pages: dict[int, list] = {}
    for placement in pdf_result.placements:
        pages.setdefault(placement.page_number, []).append(placement)

    with st.expander("Ver organização das páginas"):
        for page_number, placements in pages.items():
            st.write(f"Página {page_number}:")
            for placement in placements:
                rotation = " (girada no PDF)" if placement.rotated_on_pdf else ""
                st.write(f"- {placement.position_label.capitalize()}: {placement.output_name}{rotation}")


def _show_batch_plan_summary(results: dict) -> None:
    plan_result: FinalProcessingPlanResult | None = results.get("final_plan_result")
    if not plan_result:
        return

    final_plan = plan_result.final_plan
    st.markdown("### Planejamento do lote")
    st.write(f"Prioridade: {results.get('priority_label', 'Equilibrado')}")
    st.write(f"Organizacao usada: {final_plan.strategy}")
    if final_plan.source == "ai_validated":
        st.info("IA de lote usada e validada pelo codigo local.")
    elif final_plan.source == "fallback":
        st.info("A IA nao respondeu ou foi descartada, entao usei o planejamento local.")
    else:
        st.info("Planejamento local usado.")

    border_decisions = sum(1 for decision in final_plan.image_decisions if decision.use_borders)
    crop_decisions = sum(1 for decision in final_plan.image_decisions if decision.crop_allowed)
    rotated_decisions = sum(1 for decision in final_plan.image_decisions if decision.rotate_on_pdf)
    extra_pages = sum(1 for decision in final_plan.image_decisions if decision.create_extra_page)
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Corte seguro", crop_decisions)
    col_b.metric("Com bordas", border_decisions)
    col_c.metric("Giradas no PDF", rotated_decisions)
    col_d.metric("Pagina extra", extra_pages)

    if plan_result.corrected:
        st.info("O plano da IA foi ajustado pelas regras de seguranca locais.")
    if plan_result.ai_error and results.get("use_ai_batch_planning"):
        st.warning("A IA nao respondeu, entao usei o planejamento local.")

    with st.expander("Resumo por pagina"):
        for page in final_plan.pages:
            st.write(f"Pagina {page.page_number}:")
            for slot in page.slots:
                rotation = " (girada no PDF)" if slot.rotate_on_pdf else ""
                st.write(f"- {slot.position}: {slot.image_name}{rotation}")

    if results.get("show_batch_plan_technical"):
        with st.expander("Plano tecnico do lote"):
            st.json(plan_result.technical_payload())


def _show_local_analysis(processed_images: list, show_ai_technical: bool = False) -> None:
    analyzed_items = [item for item in processed_images if item.analysis_report]
    if not analyzed_items:
        return

    with st.expander("Análise técnica"):
        summary_tab, json_tab = st.tabs(["Resumo", "JSON"])

        with summary_tab:
            for item in analyzed_items:
                report = item.analysis_report
                st.markdown(f"**{item.original_name}**")
                st.write(f"- Orientação: {report.orientation}")
                st.write(
                    f"- Corte necessário: {report.required_crop_percent:.1f}% "
                    f"({report.required_crop_axis or 'nenhum'})"
                )
                st.write(f"- Rostos detectados: {report.faces_detected}")
                st.write(f"- Pessoas detectadas: {report.persons_detected}")
                st.write(f"- Assunto principal: {report.primary_subject_type}")
                st.write(
                    f"- Fundo sobrando: {report.background_waste_score:.2f}; "
                    f"pode aproximar: {'sim' if report.can_tighten_frame else 'nao'}"
                )
                st.write(f"- Texto detectado: {'sim' if report.text_detected else 'não'}")
                st.write(f"- Risco: {report.risk_level}")
                st.write(f"- Estratégia sugerida: {report.suggested_strategy}")
                if report.subject_crop_reason:
                    st.write(f"- Enquadramento: {report.subject_crop_reason}")
                if report.reasons:
                    st.write("- Motivos:")
                    for reason in report.reasons:
                        st.write(f"  - {reason}")

        with json_tab:
            st.json([item.analysis_report.to_dict() for item in analyzed_items])

    if show_ai_technical:
        ai_items = [item for item in processed_images if item.ai_decision or item.ai_error or item.ai_report_payload]
        if ai_items:
            with st.expander("Resposta da IA"):
                for item in ai_items:
                    st.markdown(f"**{item.original_name}**")
                    if item.ai_error:
                        st.write(f"Status: {item.ai_error}")
                    if item.final_decision_strategy:
                        st.write(f"Decisao final: {item.final_decision_strategy}")
                    if item.ai_decision:
                        st.write("Decisao validada:")
                        st.json(item.ai_decision.to_dict())
                    if item.ai_report_payload:
                        st.write("Relatorio enviado:")
                        st.json(item.ai_report_payload)
                    if item.ai_prompt:
                        st.write("Prompt enviado:")
                        st.code(item.ai_prompt, language="text")
                    if item.ai_raw_response_text:
                        st.write("Resposta bruta:")
                        st.code(item.ai_raw_response_text, language="json")


def _show_downloads(results: dict) -> None:
    st.success("Fotos processadas com sucesso.")

    processed_images = results["processed_images"]
    pdf_result = results.get("pdf_result")
    border_count = sum(1 for item in processed_images if item.used_borders)
    safe_crop_count = sum(1 for item in processed_images if item.used_safe_crop and not item.used_subject_focused_crop)
    subject_crop_count = sum(1 for item in processed_images if item.used_subject_focused_crop)
    group_count = sum(
        1
        for item in processed_images
        if getattr(item.analysis_report, "primary_subject_type", "") == "group_people"
    )
    pet_subject_count = sum(
        1
        for item in processed_images
        if getattr(item.analysis_report, "primary_subject_type", "") == "people_with_pet"
    )
    tightened_count = sum(
        1
        for item in processed_images
        if item.used_subject_focused_crop and getattr(item.analysis_report, "can_tighten_frame", False)
    )
    local_analysis_count = sum(1 for item in processed_images if item.analysis_report)
    ai_decision_count = sum(1 for item in processed_images if item.ai_decision)
    if local_analysis_count:
        st.info("Análise local ativada.")
    if ai_decision_count:
        st.info("A IA ajudou na decisao e o app validou a sugestao localmente.")
    if border_count:
        st.info("Usei bordas em algumas fotos para evitar cortes ou preservar melhor a imagem.")
    if safe_crop_count:
        st.info(f"{safe_crop_count} foto(s) foram preenchidas com corte seguro.")
    if subject_crop_count:
        st.info(f"{subject_crop_count} foto(s) usaram foco nas pessoas.")
    if pet_subject_count:
        st.info(f"{pet_subject_count} foto(s) preservaram pet junto ao grupo.")

    multiple_faces_count = sum(1 for item in processed_images if item.multiple_faces_detected)
    vertical_count = sum(1 for item in processed_images if item.original_orientation == ORIENTATION_VERTICAL)
    horizontal_count = sum(1 for item in processed_images if item.original_orientation == ORIENTATION_HORIZONTAL)
    square_count = sum(1 for item in processed_images if item.original_orientation == ORIENTATION_SQUARE)
    rotated_pdf_count = sum(1 for item in processed_images if item.rotated_on_pdf)

    st.markdown("### Resumo")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Verticais", vertical_count)
    col_b.metric("Horizontais", horizontal_count)
    col_c.metric("Quase quadradas", square_count)

    col_d, col_e, col_f = st.columns(3)
    col_d.metric("Com bordas", border_count)
    col_e.metric("Corte seguro", safe_crop_count)
    col_f.metric("Várias pessoas", multiple_faces_count)

    col_subject, col_group, col_pet = st.columns(3)
    col_subject.metric("Foco nas pessoas", subject_crop_count)
    col_group.metric("Grupos", group_count)
    col_pet.metric("Pet no assunto", pet_subject_count)
    if tightened_count:
        st.caption(f"{tightened_count} foto(s) tinham fundo sobrando e foram aproximadas.")

    if pdf_result:
        col_g, col_h = st.columns(2)
        page_count = len({placement.page_number for placement in pdf_result.placements})
        col_g.metric("Páginas PDF", page_count)
        col_h.metric("Giradas só no PDF", rotated_pdf_count)

    warnings = [item for item in processed_images if item.warning]
    if warnings:
        with st.expander("Avisos das fotos"):
            for item in warnings:
                st.write(f"{item.original_name}: {item.warning}")

    _show_batch_plan_summary(results)

    st.markdown("### Downloads")
    col_zip, col_pdf = st.columns(2)

    with col_zip:
        if not results.get("want_zip"):
            st.caption("ZIP não foi solicitado.")
        elif results.get("zip_bytes"):
            st.download_button(
                "Baixar fotos em ZIP",
                data=results["zip_bytes"],
                file_name="fotos_10x15.zip",
                mime="application/zip",
                use_container_width=True,
            )
        else:
            st.warning("Não consegui gerar o ZIP, mas você ainda pode tentar gerar novamente.")

    with col_pdf:
        if not results.get("want_pdf"):
            st.caption("PDF não foi solicitado.")
        elif results.get("pdf_bytes"):
            st.download_button(
                "Baixar PDF A4",
                data=results["pdf_bytes"],
                file_name="fotos_A4_organizadas.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning("Não consegui gerar o PDF, mas o ZIP pode ser baixado se foi criado.")

    _show_pdf_summary(pdf_result)

    st.markdown("### Prévia")
    preview_items = processed_images[:12]
    columns = st.columns(3)
    for position, item in enumerate(preview_items):
        with columns[position % 3]:
            st.image(
                generate_preview_image(item.image),
                caption=f"{item.output_name}",
                use_container_width=True,
            )

    if len(processed_images) > len(preview_items):
        st.caption(f"Mostrei as primeiras {len(preview_items)} fotos. O download inclui todas.")

    with st.expander("Detalhes do processamento"):
        for item in processed_images:
            item_reason = item.subject_crop_reason or item.safe_crop_reason
            reason = f"; {item_reason}" if item_reason else ""
            pdf_rotation = ""
            if item.pdf_layout_name:
                rotation_reason = f"; motivo: {item.pdf_rotation_reason}" if item.pdf_rotation_reason else ""
                pdf_rotation = (
                    f"; PDF: {item.pdf_layout_name}; rotacionada no PDF: "
                    f"{'sim' if item.rotated_on_pdf else 'nao'}; "
                    f"graus: {item.pdf_rotation_degrees}{rotation_reason}"
                )
            st.write(
                f"{item.original_name} -> {item.target_format}; "
                f"{item.resize_mode_used}; rostos detectados: {item.faces_detected}{reason}{pdf_rotation}"
            )

    _show_local_analysis(processed_images, bool(results.get("show_ai_technical")))


def main() -> None:
    st.set_page_config(page_title="Foto 10x15 Fácil", layout="centered")

    st.title("Foto 10x15 Fácil")
    st.caption("Envie várias fotos e gere imagens prontas para impressão, sem distorcer.")
    st.info("As fotos são processadas apenas neste computador. A ferramenta não envia imagens para a internet.")

    st.markdown("### Etapa 1 - Enviar fotos")
    uploaded_files = st.file_uploader(
        "Selecione suas fotos",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    upload_signature = tuple(
        (uploaded_file.name, getattr(uploaded_file, "size", None))
        for uploaded_file in (uploaded_files or [])
    )
    _reset_results_if_needed(upload_signature)

    st.markdown("### Etapa 2 - O que deseja gerar?")
    generate_label = st.radio(
        "Escolha a saída",
        list(GENERATE_OPTIONS.keys()),
        index=0,
        horizontal=True,
    )

    st.markdown("### Etapa 3 - Escolher ajustes")
    col_format, col_resize = st.columns(2)
    with col_format:
        target_label = st.selectbox(
            "Formato das imagens",
            list(FORMAT_OPTIONS.keys()),
            index=0,
            help="No modo automático, fotos verticais viram 10x15 e fotos horizontais viram 15x10.",
        )
    with col_resize:
        resize_label = st.selectbox(
            "Modo de ajuste",
            list(RESIZE_OPTIONS.keys()),
            index=0,
        )
    selected_resize_mode = RESIZE_OPTIONS[resize_label]
    if selected_resize_mode == RESIZE_SAFE_CROP:
        st.caption("Esse modo tenta preencher a foto sem bordas brancas, mas só corta quando o corte parece seguro.")
    elif selected_resize_mode == RESIZE_SMART:
        st.caption("O modo automático tenta cortar levemente quando não prejudica a imagem. Se houver risco, ele usa bordas.")

    frame_priority_label = st.selectbox(
        "Prioridade do enquadramento",
        list(FRAME_PRIORITY_OPTIONS.keys()),
        index=1,
        help="Equilibrado tenta cortar fundo irrelevante quando for seguro.",
    )
    if FRAME_PRIORITY_OPTIONS[frame_priority_label] == FRAME_PRIORITY_PEOPLE:
        st.info(
            "Esse modo pode cortar parede, chao, sofa, mesa ou espaco vazio "
            "para deixar as pessoas mais proximas, sem cortar ninguem."
        )
    else:
        st.caption("Foco nas pessoas tenta aproximar o grupo principal e cortar fundo irrelevante.")

    avoid_cutting_people = st.checkbox("Evitar cortar pessoas", value=True)
    st.caption("Quando essa opção está ativada, a ferramenta tenta manter uma ou várias pessoas dentro do enquadramento.")

    use_local_analysis = st.checkbox(
        "Usar análise local inteligente",
        value=True,
        help=(
            "A ferramenta usa recursos locais como detecção de rostos, pessoas, texto e bordas "
            "para evitar cortes ruins."
        ),
    )
    if use_local_analysis:
        st.caption("Análise local ativada.")

    st.markdown("### Assistente de IA")
    use_ai_decision = st.checkbox(
        "Usar IA para ajudar na decisao de corte",
        value=AI_DECISION_ENABLED_DEFAULT,
        help=(
            "A IA nao recebe a foto. Ela recebe apenas um relatorio tecnico gerado localmente, "
            "como orientacao, rostos detectados, corte necessario e risco."
        ),
    )
    if use_ai_decision:
        st.caption("A IA e opcional. Se ela nao responder, a ferramenta usa a analise local.")

    st.markdown("### Planejamento do lote")
    priority_label = st.selectbox(
        "Prioridade do processamento",
        list(PRIORITY_OPTIONS.keys()),
        index=0,
        help=(
            "Equilibrado tenta cortar levemente quando seguro. Mais seguro usa mais bordas. "
            "Economizar papel aproveita melhor o PDF. Preencher mais a foto tenta reduzir bordas."
        ),
    )
    use_ai_batch_planning = st.checkbox(
        "Usar IA para planejar o lote inteiro",
        value=False,
        help=(
            "A IA recebe apenas um resumo tecnico das fotos e sugere a melhor organizacao do PDF. "
            "As decisoes sao validadas pelo codigo antes de aplicar."
        ),
    )
    if use_ai_batch_planning:
        st.caption("A IA de lote e opcional. Se ela falhar, o app usa o planejamento local.")

    with st.expander("Plano tecnico do lote", expanded=False):
        show_batch_plan_technical = st.checkbox("Mostrar plano da IA", value=False)

    with st.expander("Opções avançadas de corte", expanded=False):
        max_safe_crop_percent = st.select_slider(
            "Máximo de corte permitido",
            options=[5, 8, 10, 12, 15, 20],
            value=12,
            help="Define quanto a ferramenta pode cortar para preencher a foto sem bordas.",
        )
        strict_people_safety = st.checkbox("Ser mais cuidadoso com fotos de pessoas", value=True)
        prefer_borders_when_uncertain = st.checkbox("Usar bordas se houver dúvida", value=True)
        avoid_cutting_text_or_objects = st.checkbox("Evitar cortar texto ou objetos nas bordas", value=True)
        show_ai_technical = st.checkbox("Mostrar resposta tecnica da IA", value=False)

    should_generate_pdf = GENERATE_OPTIONS[generate_label] in {"both", "pdf"}
    should_generate_zip = GENERATE_OPTIONS[generate_label] in {"both", "images"}

    pdf_label = list(PDF_OPTIONS.keys())[0]
    organize_label = list(PDF_ORGANIZE_OPTIONS.keys())[0]
    show_cut_lines = True
    gap_cm = PDF_PHOTO_GAP_CM

    if should_generate_pdf:
        st.markdown("### Etapa 4 - Montar PDF A4")
        pdf_label = st.selectbox(
            "Layout do PDF A4",
            list(PDF_OPTIONS.keys()),
            index=0,
        )
        selected_pdf_layout = PDF_OPTIONS[pdf_label]
        if selected_pdf_layout == PDF_LAYOUT_3_REAL_PHOTOS:
            st.caption(
                "Usa margem de 3 mm nas bordas da folha e 2 mm entre as fotos "
                "para caber 3 fotos em tamanho real."
            )
            st.write("Foto de cima: 15x10 cm.")
            st.write("Fotos de baixo: 10x15 cm.")
            st.write("Margem da folha: 3 mm.")
            st.write("Distância entre fotos: 2 mm.")
        elif selected_pdf_layout == PDF_LAYOUT_4_REAL_PHOTOS:
            st.caption(
                "Usa 4 imagens por folha A4, cada uma com 10 cm x 14,52 cm, "
                "com 3 mm de margem nas bordas da folha."
            )
            st.write("O espaco entre as imagens e calculado automaticamente com base na sobra real da folha.")
            st.write("Margem da folha: 3 mm.")
            st.write("Espaco horizontal entre colunas: 4 mm.")
            st.write("Espaco vertical entre linhas: 0,6 mm.")
            st.write("Neste layout, fotos horizontais sao giradas automaticamente no PDF para caber nos espacos verticais.")
        elif selected_pdf_layout == PDF_LAYOUT_3_PER_A4:
            st.caption(
                "Proporção 10x15: mantém o formato de foto 10x15, mas pode ficar um pouco menor "
                "para caber melhor na folha A4."
            )
        else:
            st.caption("10x15 real: mantém o tamanho físico exato da foto, mas cabe menos foto por folha.")

        organize_label = st.selectbox(
            "Organização do PDF",
            list(PDF_ORGANIZE_OPTIONS.keys()),
            index=0,
        )
        st.caption(
            "Na organização automática, fotos horizontais vão para espaços horizontais "
            "e fotos verticais vão para espaços verticais sempre que possível."
        )

        col_lines, col_gap = st.columns(2)
        with col_lines:
            show_cut_lines = st.checkbox("Mostrar linhas de corte", value=True)
        with col_gap:
            if selected_pdf_layout == PDF_LAYOUT_3_REAL_PHOTOS:
                gap_cm = PDF_PHOTO_GAP_CM
                st.caption("Espaço fixo entre fotos: 0,2 cm.")
            elif selected_pdf_layout == PDF_LAYOUT_4_REAL_PHOTOS:
                gap_cm = PDF_4UP_HORIZONTAL_GAP_CM
                st.caption(
                    f"Espacos fixos calculados: {PDF_4UP_HORIZONTAL_GAP_CM:.2f} cm horizontal "
                    f"e {PDF_4UP_VERTICAL_GAP_CM:.2f} cm vertical."
                )
            else:
                gap_cm = st.slider("Espaço entre fotos (cm)", 0.2, 1.0, 0.4, 0.1)

    if not uploaded_files:
        st.warning("Envie pelo menos uma foto para começar.")

    generate = st.button(
        "Gerar fotos",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if generate:
        processing_options = ProcessingOptions(
            target_format=FORMAT_OPTIONS[target_label],
            resize_mode=RESIZE_OPTIONS[resize_label],
            avoid_cutting_people=avoid_cutting_people,
            max_safe_crop_percent=float(max_safe_crop_percent),
            prefer_borders_when_uncertain=prefer_borders_when_uncertain,
            avoid_cutting_text_or_objects=avoid_cutting_text_or_objects,
            strict_people_safety=strict_people_safety,
            safe_crop_enabled=True,
            use_local_analysis=use_local_analysis or use_ai_decision or use_ai_batch_planning,
            use_ai_decision=use_ai_decision,
            frame_priority=FRAME_PRIORITY_OPTIONS[frame_priority_label],
        )
        pdf_options = PdfOptions(
            layout_mode=PDF_OPTIONS[pdf_label],
            show_cut_lines=show_cut_lines,
            margin_cm=(
                PDF_EDGE_MARGIN_CM
                if PDF_OPTIONS[pdf_label] in {PDF_LAYOUT_3_REAL_PHOTOS, PDF_LAYOUT_4_REAL_PHOTOS}
                else PAGE_MARGIN_CM
            ),
            gap_cm=gap_cm,
            organize_mode=PDF_ORGANIZE_OPTIONS[organize_label],
        )
        user_preferences = {
            "priority_mode": PRIORITY_OPTIONS[priority_label],
            "priority_label": priority_label,
            "pdf_organization_mode": pdf_options.organize_mode,
            "pdf_layout_mode": pdf_options.layout_mode,
            "pdf_layout_label": PDF_LAYOUT_LABELS[pdf_options.layout_mode],
            "avoid_cutting_people": avoid_cutting_people,
            "avoid_cutting_text": avoid_cutting_text_or_objects,
            "max_safe_crop_percent": float(max_safe_crop_percent),
            "show_cut_lines": show_cut_lines,
            "use_ai_individual_decision": use_ai_decision,
            "use_ai_batch_planning": use_ai_batch_planning,
            "frame_priority": FRAME_PRIORITY_OPTIONS[frame_priority_label],
            "preserve_upload_order": pdf_options.organize_mode == PDF_ORGANIZE_UPLOAD_ORDER,
        }

        processed_images = []
        failed_count = 0

        with st.spinner("Processando fotos..."):
            for index, uploaded_file in enumerate(uploaded_files, start=1):
                try:
                    original_image = open_image_safely(uploaded_file)
                    processed = process_image(
                        original_image,
                        uploaded_file.name,
                        index,
                        processing_options,
                    )
                    processed_images.append(processed)
                except Exception:
                    logger.exception("Nao consegui processar a foto %s", getattr(uploaded_file, "name", "sem_nome"))
                    failed_count += 1

        if failed_count == 1:
            st.warning("Uma das imagens não pôde ser lida.")
        elif failed_count > 1:
            st.warning("Algumas imagens não puderam ser lidas.")

        if not processed_images:
            st.error("Não consegui processar as imagens. Tente usar arquivos JPG ou PNG.")
            return

        final_plan_result = None
        try:
            final_plan_result = build_final_processing_plan(
                processed_images,
                batch_report=None,
                user_preferences=user_preferences,
                use_ai_batch_planning=use_ai_batch_planning,
            )
        except Exception:
            logger.exception("Falha no planejamento do lote")
            st.warning("Nao consegui criar o planejamento do lote. Usei a organizacao antiga do app.")

        zip_bytes = None
        pdf_bytes = None
        pdf_result = None

        if should_generate_zip:
            try:
                zip_bytes = generate_zip_bytes(processed_images)
            except Exception:
                logger.exception("Falha ao gerar ZIP")
                st.warning("Não consegui gerar o ZIP. O PDF ainda pode ser baixado se foi criado.")

        if should_generate_pdf:
            try:
                pdf_result = generate_pdf_with_summary(
                    processed_images,
                    pdf_options,
                    final_plan_result.final_plan if final_plan_result else None,
                )
                pdf_bytes = pdf_result.pdf_bytes
            except Exception:
                logger.exception("Falha ao gerar PDF")
                st.warning("Não consegui gerar o PDF A4. O ZIP ainda pode ser baixado se foi criado.")

        if pdf_bytes:
            page_count = len({placement.page_number for placement in (pdf_result.placements if pdf_result else [])})
            st.success(f"PDF A4 gerado com sucesso com {page_count} página(s).")

        st.session_state["results"] = {
            "processed_images": processed_images,
            "zip_bytes": zip_bytes,
            "pdf_bytes": pdf_bytes,
            "pdf_result": pdf_result,
            "final_plan_result": final_plan_result,
            "want_zip": should_generate_zip,
            "want_pdf": should_generate_pdf,
            "pdf_layout": PDF_LAYOUT_LABELS[pdf_options.layout_mode],
            "priority_label": priority_label,
            "use_ai_batch_planning": use_ai_batch_planning,
            "show_ai_technical": show_ai_technical,
            "show_batch_plan_technical": show_batch_plan_technical,
        }

    if st.session_state.get("results"):
        _show_downloads(st.session_state["results"])


if __name__ == "__main__":
    main()
