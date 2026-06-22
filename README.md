# Foto 10x15 Fácil

Ferramenta local para preparar várias fotos para impressão em formato 10x15 ou 15x10, sem distorcer a imagem.

Ela roda no seu computador com Python e Streamlit. As fotos não são enviadas para a internet.

## Para que serve

- Ajustar fotos individuais em proporção 10x15 vertical ou 15x10 horizontal.
- Gerar um ZIP com as imagens em JPG, 300 DPI e boa qualidade.
- Gerar um PDF A4 pronto para impressão.
- Montar páginas A4 com layouts de tamanho real ou proporção 10x15.
- Colocar 3 fotos 10x15 reais por A4 com margem de 3 mm e 2 mm entre fotos.
- Colocar 4 imagens 10x14,52 por A4 com margem de 3 mm, gap horizontal de 4 mm e gap vertical de 0,6 mm.
- Tentar detectar rostos para evitar cortes ruins.

## Instalar

No terminal, entre na pasta do projeto:

```powershell
cd c:\Projetos\PhotosFinal2
```

Crie o ambiente virtual:

```powershell
python -m venv .venv
```

Ative no Windows:

```powershell
.venv\Scripts\activate
```

Instale as dependências:

```powershell
pip install -r requirements.txt
```

## Rodar

Com o ambiente virtual ativado:

```powershell
streamlit run app.py
```

Ou:

```powershell
python -m streamlit run app.py
```

Depois, abra o endereço mostrado no terminal, normalmente:

```text
http://localhost:8501
```

## Como abrir pela janela do Windows

Também é possível abrir o app por uma janela simples de controle.

Com o ambiente virtual ativado, rode:

```powershell
python launcher.py
```

Ou dê dois cliques em:

```text
iniciar_launcher.bat
```

Na janela **Foto 10x15 Fácil**:

- **Iniciar programa** inicia o app Streamlit e abre `http://localhost:8501` no navegador.
- **Abrir no navegador** abre o endereço do app quando ele já estiver rodando.
- **Encerrar programa** fecha o Streamlit iniciado pela janela.

A janela serve apenas como painel de controle. O app principal continua sendo o Streamlit em `app.py`.

## Como usar

1. Clique em **Selecione suas fotos**.
2. Envie uma ou várias imagens JPG, JPEG ou PNG.
3. Escolha o formato das imagens: automático, 10x15 vertical ou 15x10 horizontal.
4. Escolha o modo de ajuste.
5. Escolha o layout do PDF A4.
6. Clique em **Gerar fotos**.
7. Baixe o ZIP das imagens ou o PDF A4.

## Modos de ajuste

**Automático inteligente**

É o modo padrão. A ferramenta escolhe a orientação, tenta detectar rostos e usa a lógica de corte seguro. Ela corta levemente quando isso não prejudica a imagem e usa bordas quando há risco.

**Preencher com corte seguro**

Tenta preencher a foto sem deixar bordas brancas. Só corta quando o corte parece pequeno e seguro. Se houver risco de cortar pessoas, texto, cartaz, objeto ou detalhes importantes nas bordas, a ferramenta usa bordas brancas.

**Manter foto inteira com bordas**

A foto inteira aparece. Nada é cortado. Se a proporção da foto não combinar com 10x15, o espaço que sobra fica branco.

**Preencher e cortar o mínimo necessário**

A foto preenche todo o quadro. A ferramenta mantém a proporção original e corta apenas a sobra necessária. Com a opção **Evitar cortar pessoas** ativada, ela troca para bordas quando o corte parece arriscado.

## Preencher com corte seguro

Esse modo evita bordas brancas desnecessárias, mas continua conservador.

Ele calcula quanto seria necessário cortar para preencher o formato 10x15 ou 15x10. Depois analisa as áreas que seriam removidas usando heurísticas locais com Pillow, OpenCV e NumPy. Bordas lisas, como parede, céu ou chão, tendem a ser consideradas mais seguras. Bordas com muitos detalhes, contraste, linhas ou padrões podem indicar texto, cartaz, material escolar, decoração ou objeto importante.

O modo automático inteligente usa essa mesma lógica por padrão.

Exemplos:

- Foto quase no formato 10x15: a ferramenta pode cortar um pouco e preencher sem bordas.
- Foto com grupo de pessoas perto das laterais: a ferramenta usa bordas para não cortar ninguém.
- Foto com cartaz ou texto na borda: a ferramenta evita cortar para preservar o conteúdo.

No painel **Opções avançadas de corte**, é possível ajustar o máximo de corte permitido. Valores menores são mais conservadores, especialmente para fotos com pessoas.

## Foco nas pessoas

A opção **Prioridade do enquadramento** permite escolher entre preservar mais cenário, usar o padrão equilibrado ou ativar **Foco nas pessoas**.

Esse modo identifica o assunto principal da foto, normalmente uma pessoa, um grupo, uma família, uma turma ou pessoas com pet próximo. Quando existe fundo irrelevante sobrando, ele tenta aproximar o enquadramento e cortar parede, chão, sofá, mesa, teto, céu vazio, cantos e outros espaços negativos.

O recorte continua com regras de segurança: rostos, cabeças, pessoas nas pontas, grupos, pets junto ao grupo e texto importante precisam permanecer dentro da foto. Se o recorte por assunto ficar apertado, o app abre mais o enquadramento. Se ainda houver risco, usa bordas.

Exemplo: se uma foto tem um grupo de pessoas no centro e muita parede vazia em cima, a ferramenta tenta cortar parte da parede e aproximar o grupo sem cortar ninguém.

## Análise local inteligente

Esta é a etapa 1 da inteligência local do projeto. Antes de decidir entre corte leve e bordas, a ferramenta pode gerar um relatório técnico local de cada imagem.

A análise pode usar:

- OpenCV para bordas, detalhes, variação de cor e importância das áreas que seriam cortadas.
- MediaPipe para detectar rostos, quando disponível.
- OpenCV Haar Cascade como fallback de rostos se o MediaPipe falhar.
- YOLO para detectar pessoas e objetos, quando houver um modelo local configurado.
- Tesseract para detectar regiões com texto, quando estiver instalado e configurado.
- Heurísticas locais para decidir se o corte é seguro ou se é melhor usar bordas.

Nenhuma imagem é enviada para a internet. YOLO só é usado com modelo local já existente; a ferramenta não baixa modelo automaticamente. Se MediaPipe, YOLO ou Tesseract falharem, o app continua funcionando com os recursos disponíveis.

O relatório local ajuda a preservar rostos, grupos, pessoas, texto e detalhes importantes perto das bordas. Se houver dúvida, a decisão é conservadora e usa bordas brancas.

Nesta etapa ainda não há IA de texto, API externa nem envio de relatório para IA. A etapa 2 poderá usar o relatório técnico como base para decisões mais explicáveis.

## Etapa 2 - IA por relatório técnico

A IA de texto é opcional e funciona apenas como consultora de decisão. Ela não recebe fotos, bytes, pixels, base64 nem arquivos de imagem. O app envia somente um JSON técnico gerado localmente com dados como orientação, corte necessário, rostos, pessoas, texto, bordas importantes, risco, estratégia local sugerida e um resumo numérico do assunto principal.

A IA pode sugerir uma estratégia, inclusive `subject_focused_crop`, mas o código valida tudo com regras locais rígidas. Se houver risco de cortar rosto, pessoa, grupo, pet próximo, texto ou bordas importantes, a decisão final usa bordas. Se a IA falhar, ficar sem chave, responder JSON inválido ou o endpoint não funcionar, o app continua usando a análise local da etapa 1.

Para configurar:

1. Crie um arquivo `.env` na raiz do projeto.
2. Coloque:

```env
OPENCODE_BASE_URL=https://opencode.ai/zen
OPENCODE_MODEL=minimax-m3-free
AI_DECISION_ENABLED=true
AI_DECISION_ENDPOINT_PATH=/v1/chat/completions
```

A chave da API deve ficar somente no seu `.env` local. Use `.env.example` como modelo e nunca publique o valor real.

3. Instale as dependências:

```powershell
pip install -r requirements.txt
```

4. Rode:

```powershell
streamlit run app.py
```

5. Na interface, ative **Usar IA para ajudar na decisão de corte**.

O arquivo `.env` não deve ir para o GitHub. Use `.env.example` apenas como modelo. Se o endpoint padrão não funcionar, ajuste `AI_DECISION_ENDPOINT_PATH` conforme a API do provedor.

## 10x15 real ou proporção 10x15

**10x15 real** significa que a foto impressa terá exatamente 10 cm x 15 cm, ou 15 cm x 10 cm quando horizontal. Cabe menos foto por folha, mas o tamanho físico é exato.

**Proporção 10x15** significa que a foto mantém o formato 2:3, mas pode ficar um pouco menor para caber melhor na folha A4.

No layout **3 fotos 10x15 reais por A4**, as fotos não são reduzidas para 13,9 x 9,2 cm. A foto superior fica com 15 cm x 10 cm e as duas fotos inferiores ficam com 10 cm x 15 cm.

No layout **3 fotos em proporção 10x15**, as fotos mantêm o formato, mas podem sair menores que 10x15 real.

## 3 fotos 10x15 reais por A4

Esse layout usa margem de 3 mm na borda da folha e 2 mm entre as fotos. Por isso cabem 3 fotos em tamanho real na folha A4.

- A foto de cima fica em 15x10 cm.
- As duas fotos de baixo ficam em 10x15 cm.
- As fotos não são reduzidas para 13,9 x 9,2 cm.
- Esse é o layout ideal para impressão quando o tamanho físico precisa ser respeitado.

## 4 imagens 10x14,52 por A4

Esse layout coloca 4 imagens por folha A4 em uma grade 2x2.

- Cada imagem tem 10 cm x 14,52 cm.
- A folha usa margem de 3 mm nas bordas.
- As imagens ficam encostadas nesse limite interno de 3 mm.
- O espaco entre as imagens e exatamente o que sobra da folha.
- O espaco horizontal entre as colunas fica em 4 mm.
- O espaco vertical entre as linhas fica em 0,6 mm.

Esse posicionamento e intencional e calculado fisicamente. O bloco nao e centralizado distribuindo sobra nas bordas; a sobra fica entre as imagens.

## Correcao de fotos horizontais no layout 4 imagens

No layout **4 imagens 10x14,52 por A4**, todos os espacos sao verticais. Por isso, quando uma foto horizontal e usada nesse layout, a ferramenta gira a foto 90 graus apenas no PDF.

Isso faz a imagem caber corretamente no espaco de 10 x 14,52 cm. A imagem individual no ZIP nao e alterada por causa dessa rotacao.

A decisao usa a `source_image`, que e a foto original depois da correcao de EXIF e antes de qualquer canvas com bordas. A rotacao dos pixels acontece com Pillow antes do encaixe no slot, e o layout 4 imagens usa preenchimento tipo cover por padrao para evitar bordas brancas grandes.

Essa regra vale somente para o layout 4 imagens.

- Foto vertical: entra normal.
- Foto horizontal: e girada no PDF.
- Outros layouts: continuam com suas proprias regras.

## Etapa 3 - Planejamento inteligente do lote

A ferramenta tambem pode planejar o lote inteiro antes de montar o PDF. Esse planejamento combina a analise local de cada foto, as preferencias escolhidas na interface e, opcionalmente, uma IA de texto para sugerir a organizacao das paginas.

A IA de lote e opcional. Quando ativada, ela recebe apenas um `BatchReport` em JSON com resumo tecnico das fotos. Ela nao recebe fotos, bytes, pixels, base64, caminhos locais nem arquivos de imagem.

O plano da IA pode sugerir:

- estrategia final por foto;
- qual foto vai em cada pagina;
- qual foto ocupa o slot horizontal;
- quais fotos ocupam slots verticais;
- quando rotacionar uma foto apenas no PDF;
- quando criar pagina extra;
- quando preservar ordem ou reorganizar por melhor encaixe.

O codigo valida tudo antes de aplicar. Se a IA falhar, responder JSON invalido, esquecer foto, inventar nome de arquivo ou sugerir corte perigoso, o app usa o planejamento local.

Prioridades disponiveis:

- **Mais seguro**: prefere bordas, evita cortes e preserva pessoas, texto e grupos.
- **Equilibrado**: usa corte seguro quando o risco e baixo e bordas quando ha duvida.
- **Economizar papel**: tenta aproveitar melhor as folhas, usando rotacao no PDF quando seguro.
- **Preencher mais a foto**: tenta reduzir bordas, mas ainda respeita as regras locais de seguranca.

Use IA de lote quando quiser uma segunda opiniao para organizar muitas fotos no PDF. Deixe desativada quando quiser processamento totalmente local ou quando nao tiver chave configurada. O app continua funcionando sem IA.

## PDF A4

O PDF tem quatro modos:

- **3 fotos 10x15 reais por A4**: usa A4 em retrato, margem externa de 3 mm, 2 mm entre fotos, uma foto superior em 15x10 cm e duas fotos inferiores em 10x15 cm.
- **4 imagens 10x14,52 por A4**: usa A4 em retrato, margem externa de 3 mm, 4 imagens de 10 cm x 14,52 cm, gap horizontal de 4 mm e gap vertical de 0,6 mm.
- **3 fotos em proporção 10x15**: mantém o formato 2:3, mas pode redimensionar as fotos para caber na folha.
- **2 fotos 10x15 reais**: mantém o tamanho físico real. A ferramenta pode usar página em retrato ou paisagem para fazer as fotos caberem com margem.

As linhas de corte são discretas e podem ser ativadas ou desativadas.

## Organização automática por orientação

A ferramenta detecta se cada foto está na vertical, horizontal ou quase quadrada depois de corrigir a orientação EXIF.

No formato **Automático**:

- fotos verticais viram 10x15 vertical;
- fotos horizontais viram 15x10 horizontal;
- fotos quase quadradas usam 10x15 vertical por padrão.

No PDF A4 com organização automática, fotos horizontais preferem espaços horizontais e fotos verticais preferem espaços verticais. Se não houver foto horizontal para o topo da página, uma foto vertical pode ser girada apenas no PDF para aproveitar melhor a folha.

O arquivo individual no ZIP continua correto. Uma foto vertical girada no PDF continua vertical no ZIP.

Fotos com uma ou várias pessoas continuam sendo tratadas com cuidado: a ferramenta considera o grupo inteiro e usa bordas quando o corte parece arriscado.

## Por que algumas fotos são giradas no PDF?

No layout de 3 fotos por A4, o espaço superior é horizontal. Quando só há fotos verticais, a ferramenta pode girar uma delas apenas na montagem do PDF.

Isso não altera a foto individual. Depois de recortar, basta virar a foto na mão e ela fica normal.

Esse giro serve para economizar papel sem deformar a imagem.

## 10x15 real vs proporção 10x15

**10x15 real** mantém o tamanho físico exato da foto. É ideal quando você precisa imprimir exatamente 10 cm x 15 cm ou 15 cm x 10 cm.

**Proporção 10x15** mantém o formato da foto, mas pode deixar a imagem um pouco menor para caber melhor na folha A4 com margem de segurança. Proporção 10x15 não é a mesma coisa que 10x15 real.

## Evitar cortar pessoas

Essa opção vem ativada por padrão.

Quando ela está ativada, a ferramenta usa OpenCV para detectar rostos. Se encontrar uma pessoa ou um grupo, tenta manter todos dentro do corte. Se o grupo estiver muito espalhado ou perto das bordas, a ferramenta usa bordas brancas para preservar a foto.

A regra principal é: é melhor deixar borda branca do que cortar uma pessoa.

## Qualidade da imagem

As imagens são salvas em JPG com:

- 300 DPI;
- qualidade 95;
- fundo branco quando o arquivo PNG tem transparência;
- correção automática de orientação EXIF.

A ferramenta não aplica filtros, não muda rosto, não troca fundo e não deforma pessoas.

Se uma foto tiver resolução baixa, o app mostra um aviso, mas ainda permite baixar o resultado.

## Privacidade

Todo processamento acontece localmente no computador.

O app não pede login, não usa banco de dados remoto e não envia fotos para serviços externos.

## Erros comuns

**Não consigo ativar o ambiente virtual no PowerShell**

Execute:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Depois tente ativar novamente:

```powershell
.venv\Scripts\activate
```

**O comando streamlit não foi encontrado**

Ative o ambiente virtual e instale as dependências:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

Depois rode:

```powershell
python -m streamlit run app.py
```

**Uma imagem não abriu**

Use arquivos JPG, JPEG ou PNG. Se uma foto estiver corrompida, o app ignora aquela imagem e continua com as outras.
