# Roteiro do vídeo (máximo 5 minutos)

Todos os integrantes precisam aparecer ou narrar um trecho. Gravar a demo em tela cheia,
sem cortes, com o relógio do sistema visível para provar que o fluxo roda sozinho.

## Antes de gravar

- [ ] `docker compose up --build` rodando com `PIPELINE_INTERVAL_SECONDS=60` no `.env`
- [ ] Painel aberto em http://localhost:8501 (ou na URL pública) e terminal com `docker compose logs -f scheduler`
- [ ] Uma foto **nova** do território (buraco ou trinca), ainda fora de `data/raw/`
- [ ] Webhook configurado (`WEBHOOK_URL`) e o robô RPA ou a página do webhook aberta em outra aba
- [ ] `docs/arquitetura.png` e a seção de métricas do relatório abertos

## Roteiro

| Tempo | Quem | Tela | Fala (sugestão) |
|---|---|---|---|
| 0:00–0:30 | _(nome)_ | Foto real de buraco/calçada no território | "Em _[bairro]_, um buraco só entra na fila de reparo depois que alguém reclama, e ninguém mede qual é o mais urgente. Quem mais sofre é o pedestre idoso ou com deficiência. Nosso MVP faz essa triagem sozinho, a partir de imagens da rua." |
| 0:30–1:30 | _(nome)_ | `docs/arquitetura.png` | "A cada minuto o agendador busca imagens novas — da API do Mapillary ou de uma pasta monitorada. Antes de gravar qualquer coisa, rostos e placas são desfocados. Depois vêm as transformações: letterbox, CLAHE no canal de luminância, filtro bilateral e a região de interesse. Um YOLOv8 com fine-tuning no RDD2022 detecta trincas e buracos, e o pós-processamento estima a área, calcula a severidade e registra a localização." |
| 1:30–4:00 | _(nome)_ e _(nome)_ | **Demo sem cortes** | 1) Mostrar o painel com o número atual de danos e a aba "Execuções automáticas". 2) Copiar a foto nova para `data/raw/` — "é a única ação manual, e ela simula a chegada de uma imagem; com o Mapillary nem isso é necessário". 3) Mostrar o log do agendador processando sozinho. 4) No painel, aba "Transformações": percorrer original anonimizada → letterbox → CLAHE → bilateral → ROI → baseline Canny → detecção anotada. 5) Mostrar os dados extraídos (classe, confiança, área, severidade, coordenadas). 6) Aba "Chamados": o dano de severidade alta virou chamado; mostrar o webhook/robô RPA recebendo. 7) Abrir `/docs` da API e chamar `/ranking`. |
| 4:00–4:40 | _(nome)_ | Métricas do relatório, tabela baseline × YOLO | "No conjunto de validação o modelo atingiu mAP50 de _[valor]_, com precisão _[valor]_ e recall _[valor]_. O baseline clássico, só com Canny e morfologia, acerta _[valor]_ dos danos — é o ganho do deep learning. A triagem que levava _[tempo manual]_ por imagem agora leva menos de um segundo." |
| 4:40–5:00 | _(nome)_ | Slide final com ODS e links | "Limitações: a área é uma estimativa e o modelo foi treinado com vias de outros países, então casos de severidade alta passam por validação humana. O projeto contribui para os ODS 11, 10 e 9. Próximos passos: classe de calçada quebrada com fotos nossas e medição com correção de perspectiva." |

## Depois de gravar

- [ ] Duração ≤ 5:00
- [ ] Publicar como **não listado** ou público e testar o link em aba anônima
- [ ] Colar o link no README e no relatório
