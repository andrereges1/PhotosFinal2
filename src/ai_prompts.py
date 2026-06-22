"""Prompts da IA de texto para decisao de corte."""

from __future__ import annotations

import json
from typing import Any


def build_ai_decision_prompt(report_payload: dict[str, Any]) -> str:
    report_json = json.dumps(report_payload, ensure_ascii=False, indent=2)
    return f"""
Voce e um assistente de decisao para corte e montagem de fotos 10x15.

Voce NAO recebeu a imagem.
Voce recebeu apenas um relatorio tecnico gerado por analise local.
Sua tarefa e decidir a estrategia mais segura para a foto.

Decisoes permitidas:
- safe_crop
- subject_focused_crop
- contain_with_borders
- smart_face_crop
- center_crop
- rotate_on_pdf
- create_extra_page
- manual_review

Regras obrigatorias:
1. Nunca recomende corte se houver risco de cortar rosto.
2. Nunca recomende corte se houver risco de cortar pessoa.
3. Nunca recomende corte se houver varias pessoas perto das bordas.
4. Nunca recomende corte se houver texto perto das bordas.
5. Nunca recomende corte grande.
6. Se o corte for pequeno, as bordas forem pouco importantes e pessoas/texto estiverem seguros, recomende safe_crop.
7. Se houver pessoas ou grupo bem detectado, fundo irrelevante sobrando e todas as pessoas continuarem seguras, recomende subject_focused_crop.
8. Nao recomende subject_focused_crop se pessoas estiverem perto demais das bordas, se o grupo nao couber bem, se houver texto importante na borda ou se houver duvida.
9. Se houver duvida, recomende contain_with_borders.
10. Se a foto precisa ser rotacionada apenas no PDF, recomende rotate_on_pdf.
11. Se uma foto nao encaixar bem no layout atual, recomende create_extra_page.
12. Responda apenas em JSON valido.

Formato obrigatorio da resposta:
{{
  "decision": "safe_crop",
  "confidence": 0.85,
  "reason": "O corte necessario e pequeno, os rostos estao seguros e as bordas tem baixa importancia.",
  "risk_level": "low",
  "use_borders": false,
  "allow_crop": true,
  "max_crop_percent": 8,
  "protect_faces": true,
  "protect_people": true,
  "protect_text": true,
  "rotate_on_pdf": false,
  "create_extra_page": false,
  "warnings": []
}}

Nao inclua texto fora do JSON.

Relatorio tecnico:
{report_json}
""".strip()


def build_batch_planning_prompt(batch_payload: dict[str, Any]) -> str:
    batch_json = json.dumps(batch_payload, ensure_ascii=False, indent=2)
    return f"""
Voce e um assistente de planejamento de lote para uma ferramenta de fotos 10x15.

Voce NAO recebeu imagens.
Voce recebeu apenas relatorios tecnicos em JSON.
Voce nunca deve pedir imagem, bytes, base64, pixels ou arquivos.

Sua tarefa e criar um plano para:
1. decidir estrategia final de cada foto;
2. organizar as fotos no PDF A4;
3. escolher slots adequados;
4. evitar cortes perigosos;
5. economizar papel quando possivel;
6. respeitar preferencias do usuario.

Regras obrigatorias:
- Nunca recomende distorcao.
- Nunca recomende enviar imagem para IA.
- Nunca recomende cortar rosto.
- Nunca recomende cortar pessoa.
- Nunca recomende cortar texto importante.
- Se uma foto tem risco alto, prefira bordas.
- Use subject_focused_crop quando houver pessoas ou grupo como assunto principal, fundo irrelevante sobrando e recorte seguro.
- Nao use subject_focused_crop quando pessoas, pet proximo ou texto importante possam ser cortados.
- Fotos horizontais devem preferir slots horizontais.
- Fotos verticais devem preferir slots verticais.
- No layout 4_real_images_a4, todos os slots sao verticais e fotos horizontais devem usar rotate_on_pdf.
- Fotos verticais podem ser rotacionadas apenas no PDF para ocupar slot horizontal.
- Fotos horizontais nao devem ser forcadas em slots verticais ruins.
- Se houver duvida, use bordas.
- Se o encaixe ficar ruim, crie pagina extra.
- Responda apenas JSON valido.

Estrategias globais permitidas:
- preserve_order
- best_fit
- safe_first
- paper_saving
- mixed

Layouts permitidos:
- 4_real_images_a4
- 3_photos_mixed
- 2_horizontal
- 2_real_size
- single_photo
- custom_safe

Posicoes permitidas:
- top
- top_left
- top_right
- bottom_left
- bottom_right
- center
- top_1
- top_2

Formato obrigatorio da resposta:
{{
  "strategy": "best_fit",
  "confidence": 0.82,
  "explanation": "Organizei fotos horizontais no topo e verticais nos slots inferiores, preservando imagens de maior risco com bordas.",
  "image_decisions": [
    {{
      "image_name": "foto_001_vertical_10x15.jpg",
      "decision": "safe_crop",
      "use_borders": false,
      "rotate_on_pdf": false,
      "create_extra_page": false,
      "reason": "Corte pequeno e baixo risco."
    }}
  ],
  "pages": [
    {{
      "page_number": 1,
      "layout_type": "3_photos_mixed",
      "slots": [
        {{
          "position": "top",
          "slot_type": "horizontal",
          "image_name": "foto_002_horizontal_15x10.jpg",
          "rotate_on_pdf": false,
          "fit_strategy": "normal",
          "reason": "Foto horizontal encaixa melhor no topo."
        }},
        {{
          "position": "bottom_left",
          "slot_type": "vertical",
          "image_name": "foto_001_vertical_10x15.jpg",
          "rotate_on_pdf": false,
          "fit_strategy": "safe_crop",
          "reason": "Foto vertical encaixa no slot vertical."
        }}
      ],
      "warnings": []
    }}
  ],
  "global_warnings": []
}}

Nao inclua texto fora do JSON.

BatchReport tecnico:
{batch_json}
""".strip()
