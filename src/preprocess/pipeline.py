"""Encadeia as etapas de pré-processamento e grava uma imagem por etapa."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import Paths
from src.io_utils import write_image
from src.preprocess import steps

# nome da subpasta em data/intermediate -> descrição (usada na UI e no relatório)
STEP_LABELS = {
    "00_original": "Original (já anonimizada)",
    "01_letterbox": "Letterbox 640×640",
    "02_clahe": "CLAHE no canal L (LAB)",
    "03_bilateral": "Filtro bilateral",
    "04_roi": "ROI da pista/calçada",
    "04b_perspectiva": "Correção de perspectiva",
    "05_baseline_canny": "Baseline: Canny + morfologia",
}


@dataclass
class PreprocessResult:
    letterboxed: np.ndarray
    processed: np.ndarray
    roi_mask: np.ndarray
    baseline_mask: np.ndarray
    scale: float
    pad: tuple[int, int]
    files: dict[str, Path] = field(default_factory=dict)


def _save(paths: Paths, step: str, image_id: str, img: np.ndarray) -> Path:
    return write_image(paths.intermediate / step / f"{image_id}.jpg", img)


def run_preprocess(
    img: np.ndarray,
    image_id: str,
    paths: Paths,
    size: int = 640,
    perspective: bool = False,
) -> PreprocessResult:
    """Recebe a imagem JÁ ANONIMIZADA; nada sensível é gravado em disco aqui."""
    files = {"00_original": _save(paths, "00_original", image_id, img)}

    boxed, scale, (pad_x, pad_y) = steps.letterbox(img, size)
    files["01_letterbox"] = _save(paths, "01_letterbox", image_id, boxed)

    equalized = steps.clahe_lab(boxed)
    files["02_clahe"] = _save(paths, "02_clahe", image_id, equalized)

    processed = steps.denoise_bilateral(equalized)
    files["03_bilateral"] = _save(paths, "03_bilateral", image_id, processed)

    h, w = img.shape[:2]
    content = (pad_x, pad_y, pad_x + round(w * scale), pad_y + round(h * scale))
    roi_mask = steps.road_roi_mask(processed.shape, content)
    files["04_roi"] = _save(paths, "04_roi", image_id, steps.apply_roi(processed, roi_mask))

    if perspective:
        x0, y0, x1, y1 = content
        warped = steps.warp_birdseye(processed[y0:y1, x0:x1])
        files["04b_perspectiva"] = _save(paths, "04b_perspectiva", image_id, warped)

    baseline_mask = steps.canny_morphology(processed, roi_mask)
    files["05_baseline_canny"] = _save(paths, "05_baseline_canny", image_id, baseline_mask)

    return PreprocessResult(
        letterboxed=boxed,
        processed=processed,
        roi_mask=roi_mask,
        baseline_mask=baseline_mask,
        scale=scale,
        pad=(pad_x, pad_y),
        files=files,
    )
