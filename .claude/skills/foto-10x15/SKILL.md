---
name: foto-10x15
description: Use esta skill quando for criar, corrigir, melhorar ou testar a ferramenta Foto 10x15 Fácil.
---

Você está trabalhando na ferramenta "Foto 10x15 Fácil".

Objetivo:
Criar uma ferramenta local para ajustar várias fotos em proporção 10x15 ou 15x10, sem distorcer, e gerar PDF A4 para impressão.

Regras obrigatórias:
- Nunca distorcer imagem.
- Nunca esticar largura e altura separadamente.
- Sempre manter proporção original.
- Nunca cortar rostos quando a opção de segurança estiver ativada.
- Quando houver várias pessoas, considerar o grupo inteiro.
- Se o corte for arriscado, usar bordas brancas.
- A folha A4 mede 21 x 29,7 cm.
- Nos layouts antigos, a margem externa continua sendo 1 cm quando essa for a regra do modo.
- No layout "3 fotos 10x15 reais por A4", usar margem externa de 3 mm.
- No layout "3 fotos 10x15 reais por A4", usar distância de 2 mm apenas entre fotos.
- No layout "3 fotos 10x15 reais por A4", a foto superior deve ter 15x10 cm.
- No layout "3 fotos 10x15 reais por A4", as fotos inferiores devem ter 10x15 cm.
- No layout "3 fotos 10x15 reais por A4", não usar margem de 1 cm.
- No layout "3 fotos 10x15 reais por A4", não reduzir para 13,9 x 9,2 cm.
- No layout "3 fotos 10x15 reais por A4", garantir tamanho físico real.
- No layout "4 imagens 10x14,52 por A4", usar margem externa de 3 mm.
- No layout "4 imagens 10x14,52 por A4", cada imagem mede 10 x 14,52 cm.
- No layout "4 imagens 10x14,52 por A4", as imagens devem ficar coladas ao limite interno de 3 mm das bordas.
- No layout "4 imagens 10x14,52 por A4", o espaco entre imagens deve ser calculado com base na sobra real da folha.
- No layout "4 imagens 10x14,52 por A4", o gap horizontal deve ser 4 mm e o gap vertical deve ser 0,6 mm.
- No layout "4 imagens 10x14,52 por A4", nao reduzir as imagens nem centralizar o bloco distribuindo espaco nas bordas.
- No layout "4 imagens 10x14,52 por A4", todos os slots sao verticais.
- No layout "4 imagens 10x14,52 por A4", fotos horizontais devem ser rotacionadas 90 graus apenas no PDF.
- No layout "4 imagens 10x14,52 por A4", fotos horizontais devem ser detectadas pela orientacao original corrigida por EXIF.
- No layout "4 imagens 10x14,52 por A4", detectar orientacao usando source_image corrigida por EXIF, nao a imagem final com bordas.
- No layout "4 imagens 10x14,52 por A4", rotacionar fotos horizontais antes do encaixe no slot, nunca depois de criar canvas com bordas.
- No layout "4 imagens 10x14,52 por A4", rotacionar os pixels com Pillow (`image.copy().rotate(90, expand=True)`) antes de encaixar.
- No layout "4 imagens 10x14,52 por A4", usar cover/crop como padrao para preencher o slot; nao usar contain como padrao nesse layout.
- No layout "4 imagens 10x14,52 por A4", fotos verticais entram normalmente.
- No layout "4 imagens 10x14,52 por A4", essa regra nao vale para outros layouts e nao pode alterar o ZIP.
- No layout de 3 fotos em proporção 10x15, as fotos mantêm proporção, mas podem ficar menores que 10x15 real.
- A interface deve ser simples e em português do Brasil.
- O app deve rodar localmente.
- Não enviar fotos para serviços externos.
- Gerar ZIP das imagens.
- Gerar PDF A4.
- Criar README claro.
- Atualizar requirements.txt quando necessário.

Regras de orientação:
- Detectar automaticamente fotos verticais, horizontais e quase quadradas depois de corrigir EXIF.
- No formato automático, foto vertical deve virar 10x15 vertical.
- No formato automático, foto horizontal deve virar 15x10 horizontal.
- Foto quase quadrada deve usar 10x15 vertical por padrão e mostrar aviso discreto.
- O ZIP deve preservar a orientação final correta de cada imagem individual.
- A rotação usada no PDF não deve alterar a imagem individual do ZIP.

Regras de organização do PDF:
- Organizar automaticamente pelo melhor encaixe quando essa opção estiver ativada.
- Fotos horizontais devem preferir slots horizontais.
- Fotos verticais devem preferir slots verticais.
- Fotos verticais podem ser rotacionadas apenas no PDF para ocupar o slot superior horizontal.
- Fotos horizontais não devem ser forçadas em slots verticais ruins.
- Se o encaixe ficar ruim, usar bordas, criar outra página ou usar layout horizontal seguro.
- No layout de 3 fotos em proporção 10x15, continuar respeitando margem de 1 cm, linhas de corte discretas e proporção correta.
- No layout "3 fotos 10x15 reais por A4", usar margem externa de 3 mm, 2 mm entre fotos, linhas de corte discretas e tamanho físico real.
- No layout "4 imagens 10x14,52 por A4", usar margem externa de 3 mm, slots de 10 x 14,52 cm, gap horizontal de 4 mm, gap vertical de 0,6 mm e linhas de corte discretas.
- No layout "4 imagens 10x14,52 por A4", deixar as imagens coladas ao limite interno das bordas e colocar toda a sobra entre as imagens.
- No layout "4 imagens 10x14,52 por A4", rotacionar fotos horizontais 90 graus apenas no PDF para caber nos slots verticais, sem distorcer e sem alterar o ZIP.
- No layout "4 imagens 10x14,52 por A4", a regra deve funcionar por foto em lotes mistos e nao deve decidir rotacao usando o canvas final com bordas.

Regras para pessoas:
- Nunca cortar pessoas quando a opção de segurança estiver ativada.
- Quando houver várias pessoas, considerar o grupo inteiro.
- Se rostos estiverem espalhados demais, usar bordas brancas.

Regras de corte seguro:
- Tentar evitar bordas brancas desnecessárias.
- Se o corte necessário for pequeno e seguro, aplicar corte leve.
- O modo "Automático inteligente" deve usar a lógica de corte seguro.
- O modo "Preencher com corte seguro" deve ser conservador.
- Se houver risco de cortar pessoa, rosto, grupo, texto, cartaz, material escolar, objeto ou detalhe importante, usar bordas.
- Se houver dúvida, usar bordas.
- Analisar localmente as áreas que seriam cortadas; não usar API externa nem IA online.
- Não cortar agressivamente.

Regras de recorte por assunto principal:
- Priorizar o assunto principal da foto, não o cenário inteiro.
- Na maioria das fotos, o assunto principal será pessoa, grupo de pessoas, família, turma, crianças, professores ou pessoas com pet.
- Usar a estratégia interna `subject_focused_crop` quando houver fundo irrelevante e validação local segura.
- Permitir crop assimétrico para cortar mais parede, chão, sofá, mesa, teto, céu vazio, cantos e espaço negativo.
- Não preservar cenário irrelevante sem necessidade.
- Preservar todas as pessoas, rostos, cabeças, crianças, idosos e pessoas nas pontas.
- Preservar pets próximos ao grupo.
- Preservar texto importante e objetos principais.
- Se o recorte por assunto ficar apertado, abrir mais antes de desistir.
- Se ainda houver risco, usar bordas.
- Nunca distorcer imagem.

Regras da etapa 1 de análise local:
- A etapa 1 usa análise local antes de decidir corte ou bordas.
- OpenCV analisa bordas, detalhes, variação de cor e áreas removidas pelo corte.
- MediaPipe detecta rostos quando disponível.
- OpenCV Haar Cascade é fallback de rostos se o MediaPipe falhar.
- YOLO detecta pessoas e objetos quando houver modelo local disponível.
- Tesseract detecta texto quando estiver instalado e configurado.
- Se qualquer recurso falhar, usar fallback e não quebrar o app.
- A decisão final deve ser conservadora.
- Se houver dúvida, usar bordas.
- Nunca cortar rostos.
- Nunca cortar grupo.
- Nunca cortar texto importante.
- Nunca distorcer imagens.
- Não usar IA externa nesta etapa.
- Não enviar imagem, relatório ou metadados para a internet nesta etapa.

Regras da etapa 2 de IA por relatório técnico:
- A IA de texto recebe apenas relatório técnico em JSON/texto.
- Nunca enviar imagem, bytes, base64, pixels ou arquivo de imagem para IA.
- API key deve vir de variável de ambiente ou `.env` local.
- Nunca salvar chave no código, README, logs ou arquivos versionados.
- IA é opcional.
- Se IA falhar, usar análise local.
- Toda decisão da IA deve ser validada pelo código.
- Regras de segurança locais sempre vencem a IA.
- Se houver risco de cortar rosto, pessoa, grupo ou texto, usar bordas.
- Se houver dúvida, usar bordas.
- A IA sugere; o código valida; o app decide com segurança.

Regras da etapa 3 de planejamento inteligente do lote:
- A IA pode planejar o lote inteiro.
- A IA recebe apenas BatchReport em JSON.
- Nunca enviar imagem, bytes, base64 ou pixels.
- O plano da IA precisa ser validado.
- O plano local sempre existe como fallback.
- Regras locais vencem a IA.
- Fotos horizontais preferem slots horizontais.
- Fotos verticais preferem slots verticais.
- Fotos verticais podem ser rotacionadas apenas no PDF.
- Fotos horizontais nao devem ser forcadas em slots verticais ruins.
- Se houver risco, usar bordas.
- Se houver duvida, usar bordas.
- Se a IA falhar, usar plano local.
- O PDF deve respeitar a margem definida pelo layout: 3 mm no layout "3 fotos 10x15 reais por A4" e 1 cm nos layouts que ainda usam margem antiga.
- O ZIP deve preservar orientacao correta.

Sempre que modificar o projeto:
1. Preserve simplicidade.
2. Teste importações.
3. Teste funções principais.
4. Garanta que o app rode com streamlit run app.py.
5. Não quebre recursos existentes.
6. Atualize README se mudar o uso.
7. Evite mensagens técnicas para o usuário final.
