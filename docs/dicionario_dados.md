# Dicionário de dados — `data/out/detections.parquet`

Uma linha por dano detectado. Também disponível em JSON pela API (`GET /detections`).

| Campo | Tipo | Exemplo | Descrição |
|---|---|---|---|
| `id` | string | `det_3fa9c1d20b7e_00` | Identificador do dano: `det_<imagem_id>_<índice>` |
| `imagem_id` | string | `3fa9c1d20b7e` | 12 primeiros caracteres do SHA-1 do arquivo; a mesma foto nunca é processada duas vezes |
| `captured_at` | string ISO 8601 | `2026-11-15T10:32:00-03:00` | Data da captura (sidecar → EXIF → data do arquivo) |
| `processed_at` | string ISO 8601 | `2026-11-17T08:00:04-03:00` | Quando o pipeline processou a imagem |
| `lat`, `lon` | float ou nulo | `-23.5614`, `-46.6559` | Localização (sidecar do Mapillary ou GPS do EXIF) |
| `classe` | string | `buraco` | `buraco`, `trinca` ou `calcada_quebrada` |
| `confianca` | float 0–1 | `0.87` | Confiança do modelo. No baseline clássico é uma heurística de contraste, não uma probabilidade |
| `bbox` | lista de 4 inteiros | `[212, 340, 180, 95]` | `[x, y, largura, altura]` em pixels da imagem original |
| `area_cm2` | float | `1250.0` | Área estimada do dano: caixa em pixels × fator de ocupação da classe (buraco 0,8; trinca 0,3) × (cm por pixel)². Escala padrão: `SCENE_WIDTH_CM` ÷ largura da imagem |
| `score_severidade` | float 0–1 | `0.71` | `0.5·min(área/10000, 1) + 0.2·confiança + 0.3·peso_classe` |
| `severidade` | string | `alta` | `baixa` (< 0,40), `média` (< 0,65) ou `alta` |
| `img_original` | caminho | `intermediate/00_original/<id>.jpg` | Imagem de entrada, já anonimizada |
| `img_processada` | caminho | `intermediate/03_bilateral/<id>.jpg` | Imagem após letterbox, CLAHE e filtro bilateral |
| `img_anotada` | caminho | `out/annotated/<id>.jpg` | Imagem com as caixas desenhadas |
| `fonte` | string | `mapillary` | Origem da imagem |
| `licenca` | string | `CC-BY-SA 4.0` | Licença da imagem |
| `modelo` | string | `yolo_onnx:best.onnx` | Detector que gerou a linha (`baseline_opencv` quando não há pesos treinados) |
| `regioes_anonimizadas` | inteiro | `2` | Quantos rostos/placas foram desfocados na imagem |

Caminhos são relativos à pasta `data/`. Na API, as imagens ficam em `/files/<caminho>`.

## Arquivos relacionados

| Arquivo | Conteúdo |
|---|---|
| `data/out/runs.jsonl` | Uma linha JSON por execução do agendador (início, fim, imagens, danos, chamados, erros e `imagens_removidas`: quantas imagens antigas saíram por passar de `MAX_IMAGES`) |
| `data/out/chamados/chamado_<id>.json` | Gatilho para o robô RPA: um arquivo por dano de severidade alta |
| `data/out/processed.json` | IDs das imagens já processadas (mantidos mesmo depois que a imagem é removida pelo limite) |
| `data/out/mapillary_seen.json` | IDs do Mapillary já baixados, para uma foto removida pelo limite não ser baixada de novo |
| `data/out/logs/pipeline.log` | Log textual do agendador |
