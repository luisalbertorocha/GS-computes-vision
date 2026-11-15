# Fontes de dados

Toda imagem usada no projeto está nesta tabela, com origem e licença.

| Fonte | Uso no projeto | Como obter | Licença | Observações |
|---|---|---|---|---|
| **RDD2022** — Road Damage Dataset (Arya, Maeda, Ghosh, Toshniwal, Sekimoto) | Treino e validação do detector; imagens de exemplo em `samples/` | `python -m src.model.prepare_rdd2022 --download India Japan United_States Czech` (Figshare, [doi:10.6084/m9.figshare.21431547.v1](https://doi.org/10.6084/m9.figshare.21431547.v1)) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (registro do Figshare) | Fotos de vias feitas de veículos e motos em 6 países. Rostos e placas já vêm desfocados pelos autores |
| **Mapillary** (API Graph v4) | Coleta autônoma de imagens de rua com geolocalização | `MAPILLARY_TOKEN` e `MAPILLARY_BBOX` no `.env`; o agendador baixa sozinho (ou `python -m src.ingest.fetch_mapillary`) | CC BY-SA 4.0 | O pipeline anonimiza de novo antes de gravar |
| **Fotos do grupo** | Validação no território real | Tiradas pelos integrantes em _[bairro/cidade]_ e copiadas para `data/raw/` | Próprias | Rostos e placas anonimizados antes de qualquer gravação; fotos brutas ficam fora do repositório |

Citação do RDD2022: Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Sekimoto, Y. "RDD2022: A multi-national image dataset for automatic road damage detection." *Geoscience Data Journal*, 11(4), 846–862, 2024.

## Dataset de treino

Gerado por [src/model/prepare_rdd2022.py](../src/model/prepare_rdd2022.py) em `data/dataset/` (formato YOLO, fora do git). Os números abaixo vêm de `data/dataset/stats.json`.

**Países usados:** Japão (8.662 imagens), Estados Unidos (4.805), Índia (4.510) e República Tcheca (1.573). Noruega (10 GB) e China ficaram de fora por tamanho e tempo de treino.

**Classes** (as quatro do RDD2022 reduzidas às duas do produto):

| Rótulo original | Significado | Classe do produto |
|---|---|---|
| D00 | Trinca longitudinal | `trinca` |
| D10 | Trinca transversal | `trinca` |
| D20 | Trinca em malha ("couro de jacaré") | `trinca` |
| D40 | Buraco | `buraco` |
| demais (faixa apagada, tampa, remendo) | — | ignorados |

**Divisão** — 90/10, determinística pelo hash do nome do arquivo. Só a pasta `train` do RDD2022 tem anotações, então a validação sai dela. Imagens sem dano entram como fundo, limitadas a 15% das imagens com dano.

| Split | Imagens | das quais sem dano | caixas `trinca` | caixas `buraco` |
|---|---|---|---|---|
| treino | 17.584 | 2.308 | 27.204 | 5.190 |
| validação | 1.966 | 242 | 3.093 | 572 |

**Vieses conhecidos:** nenhuma imagem é do Brasil; há 5 vezes mais trincas do que buracos; as fotos são diurnas e tiradas de dentro de veículos, não por pedestres. A classe `calcada_quebrada` do plano não existe neste dataset e depende de fotos anotadas pelo grupo.

## Imagens de exemplo (`samples/`)

Imagens da validação do RDD2022 (CC BY 4.0), usadas pelo modo demonstração do deploy público. Cada uma tem um sidecar `.json` com `fonte` e `licenca`. Não têm coordenadas, porque o RDD2022 não as publica.

## Metadados por imagem (sidecar)

Cada imagem em `data/raw/` pode ter um arquivo `.json` com o mesmo nome. A coleta do
Mapillary cria esse arquivo sozinha; para fotos próprias ele é opcional (o pipeline
usa o GPS e a data do EXIF quando existem).

```json
{
  "lat": -23.5614,
  "lon": -46.6559,
  "captured_at": "2026-11-15T10:32:00-03:00",
  "fonte": "fotos_proprias",
  "licenca": "proprio",
  "cm_per_px": 0.31
}
```

`cm_per_px` é opcional: informe quando houver um objeto de referência na foto
(ex.: régua ou folha A4 ao lado do dano) para a área em cm² ficar mais precisa.
