"""Saída estruturada: detections.parquet, estado, log de execuções e gatilho do RPA."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
import requests

from src.config import Paths

log = logging.getLogger(__name__)

COLUMNS = [
    "id",
    "imagem_id",
    "captured_at",
    "processed_at",
    "lat",
    "lon",
    "classe",
    "confianca",
    "bbox",
    "area_cm2",
    "score_severidade",
    "severidade",
    "img_original",
    "img_processada",
    "img_anotada",
    "fonte",
    "licenca",
    "modelo",
    "regioes_anonimizadas",
]


IMAGE_COLUMNS = [
    "imagem_id",
    "captured_at",
    "processed_at",
    "lat",
    "lon",
    "fonte",
    "licenca",
    "danos",
    "regioes_anonimizadas",
]


def _atomic_write(path: Path, write) -> None:
    """Grava em arquivo temporário e renomeia: a API nunca lê um arquivo pela metade."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


# ---------- detections.parquet ----------

def append_detections(paths: Paths, records: list[dict]) -> int:
    if not records:
        return 0
    new = pd.DataFrame(records, columns=COLUMNS)
    if paths.parquet.exists():
        old = pd.read_parquet(paths.parquet)
        old = old[~old["id"].isin(new["id"])]
        new = pd.concat([old, new], ignore_index=True)
    _atomic_write(paths.parquet, lambda tmp: new.to_parquet(tmp, index=False))
    return len(records)


def read_detections(paths: Paths) -> list[dict]:
    """Lê o parquet como lista de dicts prontos para JSON (sem NaN nem arrays numpy)."""
    if not paths.parquet.exists():
        return []
    df = pd.read_parquet(paths.parquet)
    records = []
    for row in df.to_dict(orient="records"):
        row["bbox"] = [int(v) for v in row["bbox"]]
        for key in ("lat", "lon"):
            if pd.isna(row[key]):
                row[key] = None
        records.append(row)
    return records


def _drop_images(file: Path, image_ids: set[str]) -> int:
    if not file.exists():
        return 0
    df = pd.read_parquet(file)
    kept = df[~df["imagem_id"].isin(image_ids)]
    if len(kept) != len(df):
        _atomic_write(file, lambda tmp: kept.to_parquet(tmp, index=False))
    return len(df) - len(kept)


def remove_images(paths: Paths, image_ids: set[str]) -> int:
    """Tira dos parquets as imagens removidas pela retenção. Retorna quantas detecções saíram."""
    if not image_ids:
        return 0
    _drop_images(paths.images, image_ids)
    return _drop_images(paths.parquet, image_ids)


# ---------- images.parquet (toda imagem analisada, com ou sem dano) ----------

def append_images(paths: Paths, records: list[dict]) -> int:
    if not records:
        return 0
    new = pd.DataFrame(records, columns=IMAGE_COLUMNS)
    if paths.images.exists():
        old = pd.read_parquet(paths.images)
        old = old[~old["imagem_id"].isin(new["imagem_id"])]
        new = pd.concat([old, new], ignore_index=True)
    _atomic_write(paths.images, lambda tmp: new.to_parquet(tmp, index=False))
    return len(records)


def read_images(paths: Paths) -> list[dict]:
    """Imagens analisadas, na ordem em que foram processadas, prontas para JSON.

    A ordem é a de gravação (as novas entram no fim), e não `processed_at`: o horário tem
    precisão de segundo e empataria entre imagens da mesma execução.
    """
    if not paths.images.exists():
        return []
    df = pd.read_parquet(paths.images)
    records = []
    for row in df.to_dict(orient="records"):
        for key in ("lat", "lon"):
            if pd.isna(row[key]):
                row[key] = None
        records.append(row)
    return records


# ---------- estado (idempotência) ----------

def load_id_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def save_id_set(path: Path, ids: set[str]) -> None:
    _atomic_write(path, lambda tmp: tmp.write_text(json.dumps(sorted(ids)), encoding="utf-8"))


def load_state(paths: Paths) -> set[str]:
    return load_id_set(paths.state)


def save_state(paths: Paths, processed: set[str]) -> None:
    save_id_set(paths.state, processed)


# ---------- log de execuções ----------

def log_run(paths: Paths, summary: dict) -> None:
    with paths.runs_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")


def first_run(paths: Paths) -> dict | None:
    """A execução mais antiga registrada: mostra desde quando existe histórico neste disco."""
    if not paths.runs_log.exists():
        return None
    with paths.runs_log.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return json.loads(line)
    return None


def read_runs(paths: Paths, limit: int = 50) -> list[dict]:
    if not paths.runs_log.exists():
        return []
    lines = paths.runs_log.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:] if line.strip()][::-1]


# ---------- gatilho para o robô RPA ----------

def ticket_payload(record: dict) -> dict:
    local = (
        f"lat {record['lat']}, lon {record['lon']}"
        if record.get("lat") is not None
        else "localização não informada"
    )
    return {
        "id": record["id"],
        "classe": record["classe"],
        "severidade": record["severidade"],
        "score_severidade": record["score_severidade"],
        "confianca": record["confianca"],
        "area_cm2": record["area_cm2"],
        "lat": record.get("lat"),
        "lon": record.get("lon"),
        "captured_at": record["captured_at"],
        "img_anotada": record["img_anotada"],
        "descricao": (
            f"Dano em via detectado automaticamente: {record['classe']}, "
            f"área estimada {record['area_cm2']:.0f} cm², severidade {record['severidade']} "
            f"({local}). Requer validação humana antes do reparo."
        ),
    }


def read_tickets(paths: Paths, limit: int = 100) -> list[dict]:
    """Chamados abertos, do mais recente para o mais antigo."""
    files = sorted(paths.tickets.glob("chamado_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    tickets = []
    for file in files[:limit]:
        try:
            tickets.append(json.loads(file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return tickets


def open_tickets(
    paths: Paths, records: list[dict], webhook_url: str | None = None, min_confidence: float = 0.0
) -> int:
    """Para cada dano de severidade alta: grava um JSON em out/chamados e chama o webhook.

    Detecções abaixo de `min_confidence` continuam no ranking, mas não abrem chamado sozinhas.
    """
    opened = 0
    for record in records:
        if record["severidade"] != "alta" or record["confianca"] < min_confidence:
            continue
        payload = ticket_payload(record)
        target = paths.tickets / f"chamado_{record['id']}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        opened += 1
        if webhook_url:
            try:
                requests.post(webhook_url, json=payload, timeout=10).raise_for_status()
            except requests.RequestException as exc:
                log.error("Webhook falhou para %s: %s", record["id"], exc)
    return opened
