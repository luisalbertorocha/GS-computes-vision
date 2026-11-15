"""Agendador do pipeline.

    python -m scheduler.run_pipeline            # uma execução (para cron / Agendador de Tarefas / Airflow)
    python -m scheduler.run_pipeline --loop     # fica rodando a cada PIPELINE_INTERVAL_SECONDS

Cada execução grava uma linha em data/out/runs.jsonl e no log data/out/logs/pipeline.log,
que servem de evidência de que o fluxo roda sem intervenção humana.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import logging
import time

from src.config import load_settings
from src.model.infer import load_detector
from src.pipeline import run_once

log = logging.getLogger("scheduler")


def release_memory() -> None:
    """Devolve ao sistema a memória que a execução usou e já liberou.

    O alocador do glibc costuma segurar essa memória no processo; em hospedagem com 512 MB
    isso soma com o painel e derruba o container. Só existe no Linux; fora dele não faz nada.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--loop", action="store_true", help="repete para sempre")
    parser.add_argument("--interval", type=int, help="segundos entre execuções (padrão: .env)")
    args = parser.parse_args()

    settings = load_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.paths.logs / "pipeline.log", encoding="utf-8"),
        ],
    )
    detector = load_detector(settings.weights, settings.img_size, settings.conf_threshold)
    interval = args.interval or settings.interval_seconds
    log.info("Detector: %s | dados: %s", detector.name, settings.paths.data)

    while True:
        summary = run_once(settings, detector)
        log.info(
            "run %s: %d coletada(s), %d amostra(s), %d processada(s), %d dano(s), %d chamado(s), "
            "%d removida(s), %d erro(s)",
            summary["run_id"],
            summary["imagens_coletadas"],
            summary["imagens_amostra"],
            summary["imagens_processadas"],
            summary["danos_detectados"],
            summary["chamados_abertos"],
            summary["imagens_removidas"],
            len(summary["erros"]),
        )
        if not args.loop:
            break
        release_memory()
        time.sleep(interval)


if __name__ == "__main__":
    main()
