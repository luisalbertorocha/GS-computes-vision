"""Coleta autônoma de imagens de rua georreferenciadas na API do Mapillary.

Precisa de MAPILLARY_TOKEN e MAPILLARY_BBOX no .env. Cada imagem baixada ganha
um sidecar .json com localização, data, fonte e licença (CC-BY-SA 4.0).

Uso manual:  python -m src.ingest.fetch_mapillary
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import requests

from src.config import Settings, load_settings
from src.postprocess import output
from src.postprocess.geo import TZ_BR

log = logging.getLogger(__name__)

API_URL = "https://graph.mapillary.com/images"
FIELDS = "id,thumb_1024_url,computed_geometry,captured_at"
# candidatas por consulta (máximo da API: 2000); só `mapillary_limit` novas são baixadas por execução.
# Valores altos fazem a busca por bbox estourar o timeout em áreas com muitas fotos.
SEARCH_LIMIT = 100
SEARCH_TIMEOUT = 60


def fetch_mapillary(settings: Settings) -> int:
    """Baixa até `mapillary_limit` imagens ainda não vistas dentro da bbox. Retorna quantas baixou."""
    if not (settings.mapillary_token and settings.mapillary_bbox):
        return 0
    # Token no cabeçalho (nunca na URL), para não aparecer em logs nem em mensagens de erro.
    response = requests.get(
        API_URL,
        params={"fields": FIELDS, "bbox": settings.mapillary_bbox, "limit": SEARCH_LIMIT},
        headers={"Authorization": f"OAuth {settings.mapillary_token}"},
        timeout=SEARCH_TIMEOUT,
    )
    if response.status_code >= 400:
        # a API devolve o motivo em JSON (token inválido, bbox maior que 0,01 grau², ...)
        try:
            reason = response.json().get("error", {}).get("message", "")
        except ValueError:
            reason = ""
        raise RuntimeError(f"Mapillary respondeu HTTP {response.status_code}. {reason}".strip())

    # ids já baixados alguma vez: a retenção apaga fotos antigas, e sem esta lista elas
    # seriam baixadas de novo a cada execução
    seen = output.load_id_set(settings.paths.mapillary_seen)
    downloaded = 0
    for item in response.json().get("data", []):
        if downloaded >= settings.mapillary_limit:
            break
        image_key = str(item["id"])
        target = settings.paths.raw / f"mapillary_{image_key}.jpg"
        url = item.get("thumb_1024_url")
        if image_key in seen or target.exists() or not url:
            continue
        image = requests.get(url, timeout=60)
        image.raise_for_status()

        meta = {"fonte": "mapillary", "licenca": "CC-BY-SA 4.0", "mapillary_id": item["id"]}
        coords = (item.get("computed_geometry") or {}).get("coordinates")
        if coords:
            meta["lon"], meta["lat"] = coords[0], coords[1]
        if item.get("captured_at"):
            captured = datetime.fromtimestamp(item["captured_at"] / 1000, TZ_BR)
            meta["captured_at"] = captured.isoformat(timespec="seconds")

        # sidecar antes da imagem: quando a imagem aparece na pasta, os metadados já existem
        target.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        target.write_bytes(image.content)
        seen.add(image_key)
        downloaded += 1
    if downloaded:
        output.save_id_set(settings.paths.mapillary_seen, seen)
    log.info("Mapillary: %d imagens novas", downloaded)
    return downloaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"{fetch_mapillary(load_settings())} imagens baixadas")
