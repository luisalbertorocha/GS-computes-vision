"""Fine-tuning do YOLO e exportação para ONNX (GPU local, Colab ou Kaggle).

    pip install -r requirements-ml.txt
    python -m src.model.prepare_rdd2022 --download India Japan United_States Czech
    python -m src.model.train --data data/dataset/data.yaml --epochs 50

Ao final copia para weights/: best.pt, best.onnx e classes.txt (nomes das classes,
na ordem do modelo), que é tudo de que o pipeline precisa para inferir sem PyTorch.
Também copia curvas, matriz de confusão e exemplos de augmentation para docs/evidencias/treino/.
Com Albumentations instalado, o Ultralytics aplica no treino desfoque, CLAHE e tons
de cinza, além de mosaico, flip horizontal e variação HSV.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_FILES = [
    "results.png",
    "results.csv",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "PR_curve.png",
    "labels.jpg",
    "train_batch0.jpg",
    "train_batch1.jpg",
    "val_batch0_labels.jpg",
    "val_batch0_pred.jpg",
]


def export_weights(best_pt: Path, weights_dir: Path, imgsz: int = 640) -> Path:
    from ultralytics import YOLO

    weights_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(best_pt))
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, device="cpu"))
    if best_pt.resolve() != (weights_dir / "best.pt").resolve():
        shutil.copy2(best_pt, weights_dir / "best.pt")
    shutil.copy2(onnx_path, weights_dir / "best.onnx")
    names = [model.names[i] for i in sorted(model.names)]
    (weights_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return weights_dir / "best.onnx"


def copy_evidence(run_dir: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in EVIDENCE_FILES:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, target / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning do detector de danos")
    parser.add_argument("--data", required=True, help="data.yaml do dataset em formato YOLO")
    parser.add_argument("--model", default="yolov8n.pt", help="pesos pré-treinados de partida")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15, help="épocas sem melhora antes de parar")
    parser.add_argument("--cache", action="store_true", help="carrega o dataset na RAM (Linux/Colab; no Windows cada worker duplica o cache e a memória estoura)")
    parser.add_argument("--name", default="zeladoria")
    parser.add_argument("--weights-dir", default=str(ROOT / "weights"))
    parser.add_argument("--evidence-dir", default=str(ROOT / "docs" / "evidencias" / "treino"))
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    result = model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        cache="ram" if args.cache else False,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
    )
    run_dir = Path(result.save_dir)
    copy_evidence(run_dir, Path(args.evidence_dir))
    print("Pesos exportados em", export_weights(run_dir / "weights" / "best.pt", Path(args.weights_dir), args.imgsz))


if __name__ == "__main__":
    main()
