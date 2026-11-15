"""Metadados de localização e data de captura.

Ordem de prioridade: arquivo sidecar `<imagem>.json` (gravado pela ingestão do
Mapillary ou à mão) -> EXIF da foto -> data de modificação do arquivo.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import ExifTags, Image

TZ_BR = timezone(timedelta(hours=-3))  # horário de Brasília


def now_iso() -> str:
    return datetime.now(TZ_BR).isoformat(timespec="seconds")


def _to_degrees(values, ref: str) -> float | None:
    try:
        d, m, s = (float(v) for v in values)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    degrees = d + m / 60 + s / 3600
    return round(-degrees if ref in ("S", "W") else degrees, 7)


def read_exif(path: Path) -> dict:
    """Extrai lat/lon e data de captura do EXIF, se existirem."""
    meta: dict = {}
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
            if gps:
                lat = _to_degrees(gps.get(2), gps.get(1, "N"))
                lon = _to_degrees(gps.get(4), gps.get(3, "E"))
                if lat is not None and lon is not None:
                    meta["lat"], meta["lon"] = lat, lon
            taken = exif.get_ifd(ExifTags.IFD.Exif).get(36867) or exif.get(306)
            if taken:
                dt = datetime.strptime(str(taken), "%Y:%m:%d %H:%M:%S").replace(tzinfo=TZ_BR)
                meta["captured_at"] = dt.isoformat(timespec="seconds")
    except Exception:  # EXIF ausente ou corrompido não pode derrubar o pipeline
        pass
    return meta


def read_sidecar(path: Path) -> dict:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def image_metadata(path: Path) -> dict:
    """Junta sidecar + EXIF + mtime. Chaves: lat, lon, captured_at, fonte, licenca, cm_per_px."""
    meta = {
        "captured_at": datetime.fromtimestamp(path.stat().st_mtime, TZ_BR).isoformat(timespec="seconds")
    }
    meta.update(read_exif(path))
    meta.update({k: v for k, v in read_sidecar(path).items() if v is not None})
    return meta
