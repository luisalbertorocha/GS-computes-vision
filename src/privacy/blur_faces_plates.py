"""Anonimização (LGPD by design): desfoca rostos e placas ANTES de qualquer gravação.

- Rostos: YuNet (rede leve do OpenCV Zoo, licença MIT, ~230 KB), que encontra
  rostos pequenos, de perfil e parcialmente cobertos. Se o modelo não estiver
  disponível, cai nos Haar cascades de rosto.
- Placas: Haar cascade que vem com o OpenCV. Erra para mais (desfoca retângulos
  que não são placas), o que é o lado seguro do erro para privacidade.

Limitações (documentar no PDF): rostos muito pequenos (< ~12 px) e placas
inclinadas podem escapar; imagens publicadas como evidência devem ser revisadas.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

Region = tuple[int, int, int, int, str]  # x, y, w, h, tipo

YUNET_MODEL = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_MAX_SIDE = 1600  # imagens maiores são reduzidas só para a detecção
FACE_SCORE = 0.5

_CASCADES: dict[str, cv2.CascadeClassifier] = {}
_CASCADE_FILES = {
    "rosto": "haarcascade_frontalface_default.xml",
    "rosto_perfil": "haarcascade_profileface.xml",
    "placa": "haarcascade_russian_plate_number.xml",
}
_yunet = None
_yunet_failed = False


def _cascade(kind: str) -> cv2.CascadeClassifier:
    if kind not in _CASCADES:
        _CASCADES[kind] = cv2.CascadeClassifier(cv2.data.haarcascades + _CASCADE_FILES[kind])
    return _CASCADES[kind]


def _haar(gray: np.ndarray, kind: str, neighbors: int, min_size: tuple[int, int]) -> list[Region]:
    found = _cascade(kind).detectMultiScale(gray, scaleFactor=1.1, minNeighbors=neighbors, minSize=min_size)
    label = "placa" if kind == "placa" else "rosto"
    return [(int(x), int(y), int(w), int(h), label) for x, y, w, h in found]


def _faces_yunet(img: np.ndarray) -> list[Region] | None:
    """Rostos pelo YuNet; None se o modelo não puder ser usado."""
    global _yunet, _yunet_failed
    if _yunet_failed:
        return None
    try:
        if _yunet is None:
            _yunet = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320), FACE_SCORE, 0.3, 5000)
        height, width = img.shape[:2]
        scale = min(1.0, YUNET_MAX_SIDE / max(height, width))
        small = img if scale == 1.0 else cv2.resize(img, (round(width * scale), round(height * scale)))
        _yunet.setInputSize((small.shape[1], small.shape[0]))
        _, faces = _yunet.detect(small)
    except (cv2.error, AttributeError) as exc:
        log.warning("YuNet indisponível (%s); usando Haar cascades para rostos.", exc)
        _yunet_failed = True
        return None
    regions: list[Region] = []
    for face in faces if faces is not None else []:
        x, y, w, h = (float(v) / scale for v in face[:4])
        regions.append((int(max(0, x)), int(max(0, y)), int(w), int(h), "rosto"))
    return regions


def find_sensitive_regions(img: np.ndarray) -> list[Region]:
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    side = min(gray.shape[:2])

    regions = _faces_yunet(img)
    if regions is None:
        min_face = (max(24, side // 40),) * 2
        regions = _haar(gray, "rosto", 5, min_face) + _haar(gray, "rosto_perfil", 5, min_face)

    regions += _haar(gray, "placa", 4, (max(30, side // 30), max(10, side // 90)))
    return regions


def blur_regions(img: np.ndarray, regions: list[Region], expand: float = 0.15) -> np.ndarray:
    out = img.copy()
    height, width = img.shape[:2]
    for x, y, w, h, _ in regions:
        dx, dy = int(w * expand), int(h * expand)
        x0, y0 = max(0, x - dx), max(0, y - dy)
        x1, y1 = min(width, x + w + dx), min(height, y + h + dy)
        if x1 <= x0 or y1 <= y0:
            continue
        k = max(15, (max(x1 - x0, y1 - y0) // 2) | 1)  # kernel ímpar proporcional à região
        out[y0:y1, x0:x1] = cv2.GaussianBlur(out[y0:y1, x0:x1], (k, k), 0)
    return out


def anonymize(img: np.ndarray) -> tuple[np.ndarray, list[Region]]:
    """Retorna (imagem com rostos/placas desfocados, regiões encontradas)."""
    regions = find_sensitive_regions(img)
    return blur_regions(img, regions), regions
