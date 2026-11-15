"""Modo demonstração: alimenta a pasta monitorada com imagens de exemplo versionadas.

Serve para o deploy público não ficar vazio quando o disco é efêmero e não há
token do Mapillary. Com DEMO_SEED_PER_RUN=N, cada execução do agendador copia até
N imagens de samples/ para data/raw/, como se tivessem acabado de chegar. As linhas
geradas ficam identificadas pela coluna `fonte` (vem do sidecar de cada amostra).
Com DEMO_SEED_PER_RUN=0 (padrão fora do container) nada é copiado.
"""
from __future__ import annotations

import shutil

from src.config import Settings
from src.ingest.watch_folder import IMAGE_EXTENSIONS


def seed_samples(settings: Settings) -> int:
    if settings.demo_seed_per_run <= 0 or not settings.samples_dir.is_dir():
        return 0
    copied = 0
    for sample in sorted(settings.samples_dir.iterdir()):
        if copied >= settings.demo_seed_per_run:
            break
        if sample.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        target = settings.paths.raw / f"amostra_{sample.name}"
        if target.exists():
            continue
        sidecar = sample.with_suffix(".json")
        if sidecar.exists():  # sidecar primeiro, como na coleta do Mapillary
            shutil.copyfile(sidecar, target.with_suffix(".json"))
        shutil.copyfile(sample, target)
        copied += 1
    return copied
