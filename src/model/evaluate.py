"""Avalia o modelo no conjunto de validação e grava as métricas para o relatório.

    python -m src.model.evaluate --data data/dataset/data.yaml

Gera docs/evidencias/metricas.json; a matriz de confusão e as curvas PR ficam
em runs/val/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação do detector de danos")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default=str(ROOT / "weights" / "best.pt"))
    parser.add_argument("--out", default=str(ROOT / "docs" / "evidencias" / "metricas.json"))
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=str(Path(args.data).resolve()), project=str(ROOT / "runs"), name="val", exist_ok=True, verbose=False
    )
    report = {
        "pesos": Path(args.weights).name,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precisao": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "por_classe": {
            model.names[int(c)]: {
                "precisao": round(float(metrics.box.p[i]), 4),
                "recall": round(float(metrics.box.r[i]), 4),
                "mAP50": round(float(metrics.box.ap50[i]), 4),
                "mAP50_95": round(float(metrics.box.ap[i]), 4),
            }
            for i, c in enumerate(metrics.box.ap_class_index)
        },
        "artefatos": "runs/val",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
