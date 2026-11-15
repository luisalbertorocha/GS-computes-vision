"""Detecção de danos.

Três detectores com a mesma interface `detect(img, baseline_mask) -> list[Detection]`,
todos recebendo a imagem letterbox (ex.: 640×640):

- YoloOnnxDetector: modelo YOLO exportado em ONNX, roda com cv2.dnn (sem PyTorch).
- YoloPtDetector:   pesos .pt via Ultralytics (precisa de requirements-ml.txt).
- BaselineDetector: Canny + morfologia; usado para comparação e como fallback
  enquanto não há pesos treinados. A "confiança" dele é uma heurística, não
  uma probabilidade.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Normaliza os rótulos do dataset para as classes do produto.
# RDD2022: D00/D10 trincas longitudinal/transversal, D20 couro de jacaré, D40 buraco.
CLASS_MAP = {
    "d00": "trinca",
    "d10": "trinca",
    "d20": "trinca",
    "d40": "buraco",
    "crack": "trinca",
    "longitudinal crack": "trinca",
    "transverse crack": "trinca",
    "alligator crack": "trinca",
    "pothole": "buraco",
    "potholes": "buraco",
    "trinca": "trinca",
    "fissura": "trinca",
    "buraco": "buraco",
    "calcada_quebrada": "calcada_quebrada",
    "broken sidewalk": "calcada_quebrada",
}


# Fração da caixa do YOLO que é dano de fato (heurística, usada na estimativa de área):
# um buraco preenche a caixa como uma elipse (~0,8); uma trinca é uma linha que ocupa
# só uma parte dela (~0,3).
BOX_FILL = {"buraco": 0.8, "trinca": 0.3}
DEFAULT_BOX_FILL = 0.6


def box_damage_area(cls: str, w: float, h: float) -> float:
    return w * h * BOX_FILL.get(cls, DEFAULT_BOX_FILL)


def normalize_class(name: str) -> str:
    return CLASS_MAP.get(name.strip().lower(), name.strip().lower().replace(" ", "_"))


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: tuple[float, float, float, float]  # x, y, w, h no espaço da imagem de entrada
    area_px: float  # área do contorno (baseline) ou caixa × fator de ocupação (YOLO)


class BaselineDetector:
    name = "baseline_opencv"

    def __init__(self, min_area_frac: float = 0.002, max_area_frac: float = 0.30, top_k: int = 5):
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.top_k = top_k

    def detect(self, img: np.ndarray, baseline_mask: np.ndarray | None = None) -> list[Detection]:
        if baseline_mask is None:
            from src.preprocess.steps import canny_morphology

            baseline_mask = canny_morphology(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        total = img.shape[0] * img.shape[1]
        contours, _ = cv2.findContours(baseline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_area_frac * total <= area <= self.max_area_frac * total):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            # trinca = forma alongada; buraco = forma compacta
            (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
            elongation = max(rw, rh) / max(1.0, min(rw, rh))
            cls = "trinca" if elongation >= 3.0 else "buraco"
            # heurística: dano costuma ser mais escuro que o entorno imediato
            inside = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(inside, [contour], -1, 255, thickness=cv2.FILLED)
            pad = max(8, int(0.25 * max(w, h)))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
            ring = np.zeros(gray.shape, dtype=np.uint8)
            ring[y0:y1, x0:x1] = 255
            ring[inside > 0] = 0
            mean_in = cv2.mean(gray, mask=inside)[0]
            mean_out = cv2.mean(gray, mask=ring)[0] or 1.0
            contrast = (mean_out - mean_in) / mean_out
            conf = float(np.clip(0.30 + contrast, 0.20, 0.75))
            detections.append(Detection(cls, round(conf, 3), (x, y, w, h), float(area)))
        detections.sort(key=lambda d: d.area_px, reverse=True)
        return detections[: self.top_k]


def _load_class_names(weights: Path) -> list[str]:
    """Nomes das classes do modelo ONNX: arquivo classes.txt ao lado dos pesos."""
    names_file = weights.parent / "classes.txt"
    if names_file.exists():
        return [line.strip() for line in names_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    log.warning("classes.txt não encontrado ao lado de %s; usando class_<i>", weights)
    return []


class YoloOnnxDetector:
    def __init__(self, weights: Path, size: int = 640, conf: float = 0.25, iou: float = 0.45):
        self.net = cv2.dnn.readNetFromONNX(str(weights))
        self.size, self.conf, self.iou = size, conf, iou
        self.names = _load_class_names(weights)
        self.name = f"yolo_onnx:{weights.name}"

    def _class_name(self, index: int) -> str:
        raw = self.names[index] if index < len(self.names) else f"class_{index}"
        return normalize_class(raw)

    def detect(self, img: np.ndarray, baseline_mask: np.ndarray | None = None) -> list[Detection]:
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()[0]
        # o blob é sempre size×size; converte de volta se a imagem tiver outro tamanho
        sx, sy = img.shape[1] / self.size, img.shape[0] / self.size

        if out.ndim == 2 and out.shape[1] == 6:
            # modelos end-to-end (NMS embutido): linhas = x1, y1, x2, y2, conf, classe
            rows = out[out[:, 4] >= self.conf]
            boxes = [[r[0], r[1], r[2] - r[0], r[3] - r[1]] for r in rows]
            confs = [float(r[4]) for r in rows]
            classes = [int(r[5]) for r in rows]
            keep = range(len(rows))
        else:
            # YOLOv8/11: (4 + nc, N) com cx, cy, w, h e um score por classe
            pred = out.T if out.shape[0] < out.shape[1] else out
            scores = pred[:, 4:]
            class_ids = scores.argmax(axis=1)
            best = scores.max(axis=1)
            mask = best >= self.conf
            pred, class_ids, best = pred[mask], class_ids[mask], best[mask]
            boxes = [[cx - w / 2, cy - h / 2, w, h] for cx, cy, w, h in pred[:, :4].tolist()]
            confs = best.tolist()
            classes = class_ids.tolist()
            keep = (
                np.array(cv2.dnn.NMSBoxes(boxes, confs, self.conf, self.iou)).flatten().tolist()
                if boxes
                else []
            )

        detections = []
        for i in keep:
            x, y, w, h = boxes[i]
            x, y, w, h = x * sx, y * sy, w * sx, h * sy
            cls = self._class_name(classes[i])
            detections.append(
                Detection(cls, round(float(confs[i]), 3), (x, y, w, h), box_damage_area(cls, w, h))
            )
        return detections


class YoloPtDetector:
    def __init__(self, weights: Path, size: int = 640, conf: float = 0.25):
        from ultralytics import YOLO  # import tardio: dependência opcional

        self.model = YOLO(str(weights))
        self.size, self.conf = size, conf
        self.name = f"yolo_pt:{weights.name}"

    def detect(self, img: np.ndarray, baseline_mask: np.ndarray | None = None) -> list[Detection]:
        result = self.model.predict(img, imgsz=self.size, conf=self.conf, verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            name = normalize_class(result.names[int(box.cls[0])])
            detections.append(
                Detection(name, round(float(box.conf[0]), 3), (x1, y1, w, h), box_damage_area(name, w, h))
            )
        return detections


def load_detector(weights: Path | None, size: int = 640, conf: float = 0.25):
    """Escolhe o detector pelos pesos disponíveis; sem pesos, usa o baseline clássico."""
    if weights is None or not Path(weights).exists():
        log.warning("Sem pesos YOLO (MODEL_WEIGHTS / weights/best.onnx): usando baseline clássico.")
        return BaselineDetector()
    weights = Path(weights)
    if weights.suffix.lower() == ".onnx":
        return YoloOnnxDetector(weights, size, conf)
    return YoloPtDetector(weights, size, conf)
