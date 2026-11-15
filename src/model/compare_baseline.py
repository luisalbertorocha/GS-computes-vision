"""Compara o baseline clássico com o YOLO nas imagens de validação, usando o código do pipeline.

    python -m src.model.compare_baseline --data data/dataset --limit 300

Para cada imagem: letterbox -> CLAHE -> bilateral (as mesmas funções do pipeline) e três
detectores: baseline (Canny + morfologia), YOLO na imagem só redimensionada e YOLO na
imagem pré-processada. Uma detecção conta como acerto quando a classe é a mesma e
IoU >= 0,5 com uma caixa anotada ainda não usada. Não precisa de PyTorch.
Grava docs/evidencias/comparacao_baseline.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.io_utils import read_image
from src.model.infer import BaselineDetector, Detection, YoloOnnxDetector
from src.preprocess import steps

ROOT = Path(__file__).resolve().parents[2]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w = min(ax + aw, bx + bw) - max(ax, bx)
    inter_h = min(ay + ah, by + bh) - max(ay, by)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    return inter / (aw * ah + bw * bh - inter)


def match(predictions: list[Detection], truth: list[tuple[str, tuple]], threshold: float = 0.5) -> tuple[int, int, int]:
    """Retorna (acertos, falsos positivos, não detectados)."""
    used: set[int] = set()
    hits = 0
    for pred in sorted(predictions, key=lambda d: d.conf, reverse=True):
        best, best_iou = None, threshold
        for index, (cls, box) in enumerate(truth):
            if index in used or cls != pred.cls:
                continue
            overlap = iou(pred.bbox, box)
            if overlap >= best_iou:
                best, best_iou = index, overlap
        if best is not None:
            used.add(best)
            hits += 1
    return hits, len(predictions) - hits, len(truth) - hits


def load_truth(label_path: Path, names: list[str], scale: float, pad: tuple[int, int], width: int, height: int):
    truth = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:])
        bw, bh = w * width * scale, h * height * scale
        x = cx * width * scale - bw / 2 + pad[0]
        y = cy * height * scale - bh / 2 + pad[1]
        truth.append((names[cls], (x, y, bw, bh)))
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline clássico × YOLO na validação")
    parser.add_argument("--data", default=str(ROOT / "data" / "dataset"))
    parser.add_argument("--weights", default=str(ROOT / "weights" / "best.onnx"))
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default=str(ROOT / "docs" / "evidencias" / "comparacao_baseline.json"))
    args = parser.parse_args()

    data = Path(args.data)
    names = (Path(args.weights).parent / "classes.txt").read_text(encoding="utf-8").split()
    images = sorted((data / "images" / "val").glob("*.jpg"))
    # amostra espalhada por todos os países (os nomes começam pelo país)
    images = images[:: max(1, len(images) // args.limit)][: args.limit]
    yolo = YoloOnnxDetector(Path(args.weights), conf=args.conf)
    baseline = BaselineDetector()

    methods = ["baseline_opencv", "yolo_letterbox", "yolo_preprocessada"]
    totals = {m: {"acertos": 0, "falsos_positivos": 0, "nao_detectados": 0, "segundos": 0.0} for m in methods}
    annotated = 0
    for image_path in images:
        img = read_image(image_path)
        label_path = data / "labels" / "val" / f"{image_path.stem}.txt"
        if img is None or not label_path.exists():
            continue
        height, width = img.shape[:2]
        boxed, scale, pad = steps.letterbox(img, 640)
        processed = steps.denoise_bilateral(steps.clahe_lab(boxed))
        content = (pad[0], pad[1], pad[0] + round(width * scale), pad[1] + round(height * scale))
        truth = load_truth(label_path, names, scale, pad, width, height)
        annotated += len(truth)

        runs = {
            "baseline_opencv": lambda: baseline.detect(
                processed, steps.canny_morphology(processed, steps.road_roi_mask(processed.shape, content))
            ),
            "yolo_letterbox": lambda: yolo.detect(boxed),
            "yolo_preprocessada": lambda: yolo.detect(processed),
        }
        for method, run in runs.items():
            start = time.perf_counter()
            predictions = run()
            totals[method]["segundos"] += time.perf_counter() - start
            hits, false_pos, missed = match(predictions, truth)
            totals[method]["acertos"] += hits
            totals[method]["falsos_positivos"] += false_pos
            totals[method]["nao_detectados"] += missed

    report = {"imagens": len(images), "danos_anotados": annotated, "iou_minimo": 0.5, "confianca_minima": args.conf, "metodos": {}}
    for method, t in totals.items():
        predicted = t["acertos"] + t["falsos_positivos"]
        report["metodos"][method] = {
            "acertos": t["acertos"],
            "falsos_positivos": t["falsos_positivos"],
            "nao_detectados": t["nao_detectados"],
            "precisao": round(t["acertos"] / predicted, 3) if predicted else 0.0,
            "recall": round(t["acertos"] / annotated, 3) if annotated else 0.0,
            "ms_por_imagem": round(1000 * t["segundos"] / max(1, len(images)), 1),
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
