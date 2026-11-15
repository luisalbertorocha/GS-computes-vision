from __future__ import annotations

import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.config import load_settings

HOLE_CENTER = (640, 520)
HOLE_AXES = (150, 80)


def make_road_image(width: int = 1280, height: int = 720) -> np.ndarray:
    """Asfalto sintético (cinza com granulação) com um buraco escuro na parte de baixo."""
    rng = np.random.default_rng(42)
    img = np.full((height, width, 3), 120, dtype=np.uint8)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    cv2.ellipse(img, HOLE_CENTER, HOLE_AXES, 0, 0, 360, (35, 35, 35), cv2.FILLED)
    return img


@pytest.fixture
def road_image() -> np.ndarray:
    return make_road_image()


@pytest.fixture
def settings(tmp_path):
    base = load_settings(tmp_path / "data")
    # nunca tocar a rede nem pesos reais durante os testes
    return replace(
        base,
        mapillary_token=None,
        mapillary_bbox=None,
        webhook_url=None,
        weights=None,
        demo_seed_per_run=0,
        samples_dir=tmp_path / "samples",
    )


@pytest.fixture
def raw_image(settings, road_image):
    """Grava a imagem sintética em data/raw com um sidecar de localização."""
    path = settings.paths.raw / "rua_teste.jpg"
    cv2.imwrite(str(path), road_image)
    path.with_suffix(".json").write_text(
        json.dumps({"lat": -23.5614, "lon": -46.6559, "fonte": "sintetico", "licenca": "teste"}),
        encoding="utf-8",
    )
    return path
