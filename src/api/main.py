"""API de leitura dos resultados.   uvicorn src.api.main:app --reload

A API só lê o que o agendador produziu; ela não processa imagens. As imagens
brutas (data/raw) nunca são servidas: só as intermediárias e anotadas, que já
passaram pela anonimização.
"""
from __future__ import annotations

from collections import Counter

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from src.config import Settings, load_settings
from src.postprocess import output
from src.postprocess.severity import (
    AREA_REF_CM2,
    CLASS_WEIGHTS,
    THRESHOLD_ALTA,
    THRESHOLD_MEDIA,
    W_AREA,
    W_CLASSE,
    W_CONF,
)
from src.preprocess.pipeline import STEP_LABELS


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    paths = settings.paths
    app = FastAPI(
        title="Zeladoria Viária · Visão Computacional",
        description="Danos em vias e calçadas detectados automaticamente a partir de imagens.",
        version="0.1.0",
    )
    app.mount("/files/intermediate", StaticFiles(directory=paths.intermediate), name="intermediate")
    app.mount("/files/out/annotated", StaticFiles(directory=paths.annotated), name="annotated")

    @app.get("/health")
    def health() -> dict:
        runs = output.read_runs(paths, limit=1)
        return {"status": "ok", "ultima_execucao": runs[0] if runs else None}

    @app.get("/detections")
    def detections(
        severidade: str | None = Query(None, description="baixa, média ou alta"),
        classe: str | None = None,
        limit: int = Query(100, ge=1, le=1000),
    ) -> list[dict]:
        rows = output.read_detections(paths)
        if severidade:
            rows = [r for r in rows if r["severidade"] == severidade]
        if classe:
            rows = [r for r in rows if r["classe"] == classe]
        rows.sort(key=lambda r: r["processed_at"], reverse=True)
        return rows[:limit]

    @app.get("/detections/{detection_id}")
    def detection(detection_id: str) -> dict:
        for row in output.read_detections(paths):
            if row["id"] == detection_id:
                return row
        raise HTTPException(status_code=404, detail="Detecção não encontrada")

    @app.get("/ranking")
    def ranking(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
        """Danos ordenados por prioridade de reparo (maior score primeiro)."""
        rows = output.read_detections(paths)
        rows.sort(key=lambda r: r["score_severidade"], reverse=True)
        return rows[:limit]

    @app.get("/stats")
    def stats() -> dict:
        rows = output.read_detections(paths)
        return {
            "total_danos": len(rows),
            "imagens_com_dano": len({r["imagem_id"] for r in rows}),
            "imagens_processadas": len(output.load_state(paths)),
            "por_classe": dict(Counter(r["classe"] for r in rows)),
            "por_severidade": dict(Counter(r["severidade"] for r in rows)),
        }

    @app.get("/steps/{imagem_id}")
    def steps(imagem_id: str) -> list[dict]:
        """Links das imagens de cada transformação intermediária de uma imagem."""
        found = []
        for step, label in STEP_LABELS.items():
            if (paths.intermediate / step / f"{imagem_id}.jpg").exists():
                found.append(
                    {"etapa": step, "descricao": label, "url": f"/files/intermediate/{step}/{imagem_id}.jpg"}
                )
        if (paths.annotated / f"{imagem_id}.jpg").exists():
            found.append(
                {
                    "etapa": "09_overlay",
                    "descricao": "Detecções anotadas",
                    "url": f"/files/out/annotated/{imagem_id}.jpg",
                }
            )
        if not found:
            raise HTTPException(status_code=404, detail="Imagem não encontrada")
        return found

    @app.get("/runs")
    def runs(limit: int = Query(20, ge=1, le=200)) -> list[dict]:
        """Histórico das execuções do agendador (evidência da automação)."""
        return output.read_runs(paths, limit)

    @app.get("/tickets")
    def tickets(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
        """Chamados gerados para danos de severidade alta (o mesmo conteúdo enviado ao webhook)."""
        return output.read_tickets(paths, limit)

    @app.get("/severity-formula")
    def severity_formula() -> dict:
        """Transparência: como o ranking é calculado."""
        return {
            "formula": "score = w_area*min(area_cm2/area_ref, 1) + w_conf*confianca + w_classe*peso_classe",
            "pesos": {"w_area": W_AREA, "w_conf": W_CONF, "w_classe": W_CLASSE},
            "area_ref_cm2": AREA_REF_CM2,
            "peso_classe": CLASS_WEIGHTS,
            "faixas": {"baixa": f"< {THRESHOLD_MEDIA}", "média": f"< {THRESHOLD_ALTA}", "alta": f">= {THRESHOLD_ALTA}"},
        }

    return app


app = create_app()
