"""Copia para docs/evidencias/ o que o relatório referencia, a partir de data/.

    python docs/collect_evidence.py                 # usa o dano de maior score
    python docs/collect_evidence.py --id det_xxx_00 # ou um dano específico

Gera:
- docs/evidencias/etapas/<etapa>.jpg   uma imagem por transformação, da mesma foto
- docs/evidencias/amostra_detections.csv e .md   amostra dos dados extraídos
- docs/evidencias/runs.jsonl           últimas execuções automáticas
- docs/evidencias/chamado_exemplo.json um chamado gerado
Depois rode docs/build_pdf.py para atualizar o PDF.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import load_settings  # noqa: E402
from src.postprocess import output  # noqa: E402
from src.preprocess.pipeline import STEP_LABELS  # noqa: E402

SAMPLE_COLUMNS = ["id", "classe", "confianca", "bbox", "area_cm2", "score_severidade", "severidade", "lat", "lon", "fonte"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reúne as evidências usadas no relatório")
    parser.add_argument("--id", help="id da detecção cuja imagem ilustra as etapas")
    parser.add_argument("--data-dir", help="pasta de dados (padrão: DATA_DIR ou data/)")
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()

    paths = load_settings(args.data_dir).paths
    rows = output.read_detections(paths)
    if not rows:
        sys.exit("Nenhuma detecção em detections.parquet. Rode o pipeline primeiro.")
    rows.sort(key=lambda r: r["score_severidade"], reverse=True)
    chosen = next((r for r in rows if r["id"] == args.id), None) if args.id else rows[0]
    if chosen is None:
        sys.exit(f"Detecção não encontrada: {args.id}")

    evidence = ROOT / "docs" / "evidencias"
    steps_dir = evidence / "etapas"
    steps_dir.mkdir(parents=True, exist_ok=True)
    for step in STEP_LABELS:
        source = paths.intermediate / step / f"{chosen['imagem_id']}.jpg"
        if source.exists():
            shutil.copyfile(source, steps_dir / f"{step}.jpg")
    shutil.copyfile(paths.data / chosen["img_anotada"], steps_dir / "09_overlay.jpg")

    # amostra: as linhas da imagem escolhida primeiro, depois as de maior score
    same_image = [r for r in rows if r["imagem_id"] == chosen["imagem_id"]]
    others = [r for r in rows if r["imagem_id"] != chosen["imagem_id"]]
    sample = pd.DataFrame((same_image + others)[: args.rows])[SAMPLE_COLUMNS]
    sample.to_csv(evidence / "amostra_detections.csv", index=False)
    header = "| " + " | ".join(SAMPLE_COLUMNS) + " |\n|" + "---|" * len(SAMPLE_COLUMNS) + "\n"
    body = "\n".join(
        "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
        for row in sample.itertuples(index=False)
    )
    (evidence / "amostra_detections.md").write_text(header + body + "\n", encoding="utf-8")

    runs = output.read_runs(paths, limit=10)[::-1]
    (evidence / "runs.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in runs) + "\n", encoding="utf-8"
    )
    tickets = output.read_tickets(paths, limit=1)
    if tickets:
        (evidence / "chamado_exemplo.json").write_text(
            json.dumps(tickets[0], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Evidências de {chosen['id']} ({chosen['classe']}, {chosen['severidade']}) em {evidence}")


if __name__ == "__main__":
    main()
