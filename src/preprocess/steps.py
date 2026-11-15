"""Etapas de pré-processamento. Uma função pura por etapa (entrada e saída em BGR uint8).

Os parâmetros padrão aqui são os que vão para o relatório de transformações.
"""
from __future__ import annotations

import cv2
import numpy as np

PAD_COLOR = (114, 114, 114)


def letterbox(img: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Etapa 1: redimensiona mantendo a proporção e completa com bordas cinza.

    Retorna (imagem size×size, escala aplicada, (pad_x, pad_y)) para permitir
    converter coordenadas de volta para a imagem original.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    out = np.full((size, size, 3), PAD_COLOR, dtype=np.uint8)
    out[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return out, scale, (pad_x, pad_y)


def clahe_lab(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Etapa 2: equalização adaptativa (CLAHE) só na luminância (canal L do LAB).

    Corrige sombra e sol forte sem alterar as cores.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def denoise_bilateral(
    img: np.ndarray, d: int = 7, sigma_color: float = 50, sigma_space: float = 50
) -> np.ndarray:
    """Etapa 3: filtro bilateral — remove granulação preservando as bordas do dano."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def road_roi_mask(
    shape: tuple[int, ...],
    content: tuple[int, int, int, int] | None = None,
    top_frac: float = 0.35,
    top_inset_frac: float = 0.15,
) -> np.ndarray:
    """Etapa 4: máscara trapezoidal da região de pista/calçada (parte inferior da cena).

    `content` = (x0, y0, x1, y1) da área útil dentro do letterbox, para que as
    bordas cinza não entrem na ROI.
    """
    h, w = shape[:2]
    x0, y0, x1, y1 = content or (0, 0, w, h)
    cw, ch = x1 - x0, y1 - y0
    margin = 3  # evita a borda entre imagem e padding
    top = y0 + int(ch * top_frac)
    inset = int(cw * top_inset_frac)
    pts = np.array(
        [
            [x0 + margin, y1 - margin],
            [x1 - margin, y1 - margin],
            [x1 - margin - inset, top],
            [x0 + margin + inset, top],
        ],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def apply_roi(img: np.ndarray, mask: np.ndarray, dim: float = 0.25) -> np.ndarray:
    """Visualização da ROI: escurece o que está fora da máscara."""
    dark = (img * dim).astype(np.uint8)
    return np.where(mask[..., None] > 0, img, dark)


def warp_birdseye(
    img: np.ndarray, top_frac: float = 0.45, top_width_frac: float = 0.4
) -> np.ndarray:
    """Etapa 4 (opcional): correção de perspectiva para vista superior aproximada.

    Assume câmera apontada para frente; o trapézio da pista vira um retângulo.
    """
    h, w = img.shape[:2]
    half_top = w * top_width_frac / 2
    src = np.float32(
        [[w / 2 - half_top, h * top_frac], [w / 2 + half_top, h * top_frac], [w, h], [0, h]]
    )
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h))


def canny_morphology(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    low: int = 50,
    high: int = 150,
    kernel: int = 9,
    blur: int = 5,
) -> np.ndarray:
    """Etapa 5 (baseline clássico): bordas de Canny + fechamento morfológico.

    Retorna uma máscara binária com as regiões candidatas a dano.
    O Canny roda na imagem inteira e a ROI é aplicada depois, para a própria
    borda da máscara não virar uma detecção.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges = cv2.Canny(gray, low, high)
    if mask is not None:
        edges = cv2.bitwise_and(edges, mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    # preenche o interior dos contornos fechados
    filled = np.zeros_like(closed)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return cv2.morphologyEx(filled, cv2.MORPH_OPEN, k)
