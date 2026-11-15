"""Pipeline de ponta a ponta para uma imagem e para uma execução agendada."""
from __future__ import annotations

import logging
import unicodedata
import uuid
from pathlib import Path

import cv2
import numpy as np

from src.config import Settings
from src.ingest.fetch_mapillary import fetch_mapillary
from src.ingest.seed_samples import seed_samples
from src.ingest.watch_folder import find_new_images
from src.io_utils import read_image, write_image
from src.model.infer import load_detector
from src.postprocess import output
from src.postprocess.geo import image_metadata, now_iso
from src.postprocess.severity import severity_label, severity_score
from src.preprocess.pipeline import run_preprocess
from src.privacy.blur_faces_plates import anonymize

log = logging.getLogger(__name__)

SEVERITY_COLORS = {"baixa": (80, 175, 76), "média": (0, 152, 255), "alta": (54, 67, 244)}  # BGR


def _ascii(text: str) -> str:
    """As fontes do cv2.putText não têm acentos."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def draw_overlay(img: np.ndarray, records: list[dict]) -> np.ndarray:
    out = img.copy()
    thickness = max(2, round(max(img.shape[:2]) / 400))
    font_scale = max(0.5, max(img.shape[:2]) / 1100)
    for record in records:
        x, y, w, h = record["bbox"]
        color = SEVERITY_COLORS[record["severidade"]]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        label = _ascii(f"{record['classe']} {record['confianca']:.2f} | {record['severidade']}")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = max(th + 8, y)
        cv2.rectangle(out, (x, top - th - 8), (x + tw + 6, top), color, cv2.FILLED)
        cv2.putText(
            out, label, (x + 3, top - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
        )
    return out


def process_image(path: Path, image_id: str, settings: Settings, detector) -> list[dict]:
    """Anonimiza -> pré-processa -> detecta -> mede -> classifica. Retorna as linhas do parquet."""
    paths = settings.paths
    raw = read_image(path)
    if raw is None:
        raise ValueError(f"Imagem ilegível: {path.name}")

    img, regions = anonymize(raw)
    del raw  # a partir daqui só existe a versão anonimizada
    pre = run_preprocess(img, image_id, paths, settings.img_size)

    # o baseline clássico sempre trabalha sobre a imagem pré-processada (é dela a máscara)
    use_processed = settings.detect_on == "processed" or detector.name == "baseline_opencv"
    detections = detector.detect(pre.processed if use_processed else pre.letterboxed, pre.baseline_mask)

    meta = image_metadata(path)
    height, width = img.shape[:2]
    cm_per_px = float(meta.get("cm_per_px") or settings.scene_width_cm / width)
    pad_x, pad_y = pre.pad
    annotated_path = paths.annotated / f"{image_id}.jpg"
    processed_at = now_iso()

    records = []
    for index, det in enumerate(detections):
        # desfaz o letterbox: coordenadas voltam para a imagem original
        x, y, w, h = det.bbox
        x0 = int(np.clip((x - pad_x) / pre.scale, 0, width - 1))
        y0 = int(np.clip((y - pad_y) / pre.scale, 0, height - 1))
        bw = int(min(w / pre.scale, width - x0))
        bh = int(min(h / pre.scale, height - y0))
        if bw < 2 or bh < 2:
            continue
        area_cm2 = round(det.area_px / pre.scale**2 * cm_per_px**2, 1)
        score = severity_score(area_cm2, det.conf, det.cls)
        records.append(
            {
                "id": f"det_{image_id}_{index:02d}",
                "imagem_id": image_id,
                "captured_at": meta["captured_at"],
                "processed_at": processed_at,
                "lat": meta.get("lat"),
                "lon": meta.get("lon"),
                "classe": det.cls,
                "confianca": det.conf,
                "bbox": [x0, y0, bw, bh],
                "area_cm2": area_cm2,
                "score_severidade": score,
                "severidade": severity_label(score),
                "img_original": paths.rel(pre.files["00_original"]),
                "img_processada": paths.rel(pre.files["03_bilateral"]),
                "img_anotada": paths.rel(annotated_path),
                "fonte": meta.get("fonte") or settings.default_source,
                "licenca": meta.get("licenca") or settings.default_license,
                "modelo": detector.name,
                "regioes_anonimizadas": len(regions),
            }
        )

    write_image(annotated_path, draw_overlay(img, records))
    return records


def run_once(settings: Settings, detector=None, trigger: str = "agendado") -> dict:
    """Uma execução completa: coleta -> processa o que é novo -> grava -> dispara chamados."""
    paths = settings.paths
    started = now_iso()
    errors: list[str] = []

    fetched = 0
    try:
        fetched = fetch_mapillary(settings)
    except Exception as exc:  # falha de rede não impede processar a pasta monitorada
        log.error("Coleta Mapillary falhou: %s", exc)
        errors.append(f"mapillary: {exc}")
    seeded = seed_samples(settings)

    processed = output.load_state(paths)
    new_images = find_new_images(paths.raw, processed)
    if new_images and detector is None:
        detector = load_detector(settings.weights, settings.img_size, settings.conf_threshold)

    records: list[dict] = []
    done = 0
    for path, image_id in new_images:
        try:
            found = process_image(path, image_id, settings, detector)
        except Exception as exc:  # não marca como processada: tenta de novo na próxima execução
            log.exception("Falha em %s", path.name)
            errors.append(f"{path.name}: {exc}")
            continue
        log.info("%s -> %d dano(s)", path.name, len(found))
        records.extend(found)
        processed.add(image_id)
        done += 1

    output.append_detections(paths, records)
    tickets = output.open_tickets(paths, records, settings.webhook_url, settings.ticket_min_confidence)
    output.save_state(paths, processed)

    summary = {
        "run_id": uuid.uuid4().hex[:8],
        "gatilho": trigger,
        "inicio": started,
        "fim": now_iso(),
        "imagens_coletadas": fetched,
        "imagens_amostra": seeded,
        "imagens_processadas": done,
        "danos_detectados": len(records),
        "chamados_abertos": tickets,
        "modelo": getattr(detector, "name", None),
        "erros": errors,
    }
    output.log_run(paths, summary)
    return summary
