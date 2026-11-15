import json
from dataclasses import replace

import pandas as pd

from src.ingest import fetch_mapillary as mapillary
from src.model.infer import BaselineDetector, Detection, normalize_class
from src.pipeline import run_once
from src.postprocess import output
from src.postprocess.severity import severity_label, severity_score
from tests.conftest import HOLE_AXES, HOLE_CENTER


def test_severity_score_and_bands():
    assert severity_score(10000, 1.0, "buraco") == 1.0
    assert severity_score(5000, 1.0, "buraco") == 0.75
    assert severity_score(0, 0.0, "trinca") == 0.12
    assert severity_score(99999, 0.5, "buraco") == severity_score(10000, 0.5, "buraco")  # satura
    assert severity_score(2000, 0.8, "buraco") > severity_score(2000, 0.8, "trinca")
    assert [severity_label(s) for s in (0.39, 0.40, 0.64, 0.65)] == ["baixa", "média", "média", "alta"]


def test_box_area_uses_class_fill_factor():
    from src.model.infer import box_damage_area

    assert box_damage_area("buraco", 100, 50) == 4000
    assert box_damage_area("trinca", 100, 50) == 1500


def test_class_names_are_normalized():
    assert normalize_class("D40") == "buraco"
    assert normalize_class("D00") == "trinca"
    assert normalize_class("Manhole Cover") == "manhole_cover"


def test_run_once_end_to_end(settings, raw_image):
    summary = run_once(settings)
    paths = settings.paths

    assert summary["imagens_processadas"] == 1 and summary["erros"] == []
    assert summary["modelo"] == "baseline_opencv"

    rows = output.read_detections(paths)
    assert len(rows) == summary["danos_detectados"] >= 1
    hole = max(rows, key=lambda r: r["area_cm2"])
    assert hole["classe"] == "buraco"
    assert (hole["lat"], hole["lon"]) == (-23.5614, -46.6559)
    assert hole["fonte"] == "sintetico" and hole["licenca"] == "teste"

    # bbox volta para coordenadas da imagem original e cobre o buraco desenhado
    x, y, w, h = hole["bbox"]
    assert abs((x + w / 2) - HOLE_CENTER[0]) < 25 and abs((y + h / 2) - HOLE_CENTER[1]) < 25
    assert abs(w - 2 * HOLE_AXES[0]) < 50 and abs(h - 2 * HOLE_AXES[1]) < 50

    # área: elipse de 300×160 px com 400 cm / 1280 px
    expected = 3.14159 * 150 * 80 * (400 / 1280) ** 2
    assert abs(hole["area_cm2"] - expected) / expected < 0.25

    # uma imagem por etapa + anotada
    for step in ("00_original", "01_letterbox", "02_clahe", "03_bilateral", "04_roi", "05_baseline_canny"):
        assert (paths.intermediate / step / f"{hole['imagem_id']}.jpg").exists()
    assert (paths.data / hole["img_anotada"]).exists()

    assert list(pd.read_parquet(paths.parquet).columns) == output.COLUMNS
    assert len(output.read_runs(paths)) == 1


def test_run_once_is_idempotent(settings, raw_image):
    first = run_once(settings)
    second = run_once(settings)
    assert second["imagens_processadas"] == 0 and second["danos_detectados"] == 0
    assert len(output.read_detections(settings.paths)) == first["danos_detectados"]
    assert len(output.read_runs(settings.paths)) == 2


def test_unreadable_image_is_reported_and_retried(settings):
    (settings.paths.raw / "quebrada.jpg").write_bytes(b"nao sou uma imagem")
    summary = run_once(settings)
    assert summary["imagens_processadas"] == 0 and len(summary["erros"]) == 1
    assert output.load_state(settings.paths) == set()


class _AlwaysHigh:
    name = "fake"

    def detect(self, img, baseline_mask=None):
        return [Detection("buraco", 0.95, (200, 300, 300, 200), 300 * 200)]


def test_high_severity_opens_ticket_and_calls_webhook(settings, raw_image, monkeypatch):
    calls = []

    class _Response:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        output.requests, "post", lambda url, json, timeout: calls.append((url, json)) or _Response()
    )
    settings = replace(settings, webhook_url="https://exemplo.test/hook")
    summary = run_once(settings, detector=_AlwaysHigh())

    assert summary["chamados_abertos"] == 1
    tickets = list(settings.paths.tickets.glob("chamado_*.json"))
    assert len(tickets) == 1
    payload = json.loads(tickets[0].read_text(encoding="utf-8"))
    assert payload["severidade"] == "alta" and payload["lat"] == -23.5614
    assert calls == [("https://exemplo.test/hook", payload)]


def test_low_confidence_damage_is_ranked_but_opens_no_ticket(settings, raw_image):
    class _Unsure:
        name = "fake"

        def detect(self, img, baseline_mask=None):
            return [Detection("buraco", 0.3, (100, 250, 400, 250), 400 * 250)]

    summary = run_once(settings, detector=_Unsure())
    rows = output.read_detections(settings.paths)
    assert [r["severidade"] for r in rows] == ["alta"]
    assert summary["chamados_abertos"] == 0 and not list(settings.paths.tickets.glob("*.json"))


def test_baseline_ignores_tiny_blobs(road_image):
    import numpy as np

    mask = np.zeros((640, 640), np.uint8)
    mask[10:14, 10:14] = 255
    assert BaselineDetector().detect(np.zeros((640, 640, 3), np.uint8), mask) == []


def test_demo_mode_drips_samples_into_the_watched_folder(settings, road_image):
    import cv2

    settings.samples_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        cv2.imwrite(str(settings.samples_dir / name), road_image + (ord(name[0]) % 7))
        (settings.samples_dir / name).with_suffix(".json").write_text(
            json.dumps({"fonte": "amostra_teste", "licenca": "CC BY 4.0"}), encoding="utf-8"
        )

    assert run_once(settings)["imagens_amostra"] == 0  # desligado por padrão

    demo = replace(settings, demo_seed_per_run=2)
    first, second, third = run_once(demo), run_once(demo), run_once(demo)
    assert [r["imagens_amostra"] for r in (first, second, third)] == [2, 1, 0]
    assert first["imagens_processadas"] == 2 and second["imagens_processadas"] == 1
    assert {r["fonte"] for r in output.read_detections(settings.paths)} == {"amostra_teste"}


def test_mapillary_error_never_leaks_the_token(settings, monkeypatch):
    settings = replace(settings, mapillary_token="segredo123", mapillary_bbox="0,0,1,1")

    class _Denied:
        status_code = 401

        def json(self):
            return {"error": {"message": "Invalid OAuth access token"}}

    monkeypatch.setattr(mapillary.requests, "get", lambda url, **kwargs: _Denied())
    summary = run_once(settings)
    assert len(summary["erros"]) == 1 and "401" in summary["erros"][0]
    assert "segredo123" not in json.dumps(summary)


def test_fetch_mapillary_downloads_only_new_images(settings, monkeypatch):
    settings = replace(settings, mapillary_token="tok", mapillary_bbox="-46.66,-23.57,-46.65,-23.56", mapillary_limit=2)
    items = [
        {
            "id": str(i),
            "thumb_1024_url": f"https://img.test/{i}.jpg",
            "computed_geometry": {"type": "Point", "coordinates": [-46.65 - i / 1000, -23.56]},
            "captured_at": 1794000000000,
        }
        for i in range(3)
    ]

    class _Response:
        status_code = 200

        def __init__(self, payload=None, content=b""):
            self._payload, self.content = payload, content

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def fake_get(url, **kwargs):
        if url == mapillary.API_URL:
            assert kwargs["headers"]["Authorization"] == "OAuth tok"
            return _Response({"data": items})
        return _Response(content=b"jpeg-bytes")

    monkeypatch.setattr(mapillary.requests, "get", fake_get)

    assert mapillary.fetch_mapillary(settings) == 2  # respeita o limite por execução
    assert mapillary.fetch_mapillary(settings) == 1  # só a que faltava
    assert mapillary.fetch_mapillary(settings) == 0
    meta = json.loads((settings.paths.raw / "mapillary_1.json").read_text(encoding="utf-8"))
    assert abs(meta["lon"] + 46.651) < 1e-9 and meta["lat"] == -23.56
    assert meta["licenca"] == "CC-BY-SA 4.0" and meta["captured_at"].endswith("-03:00")
