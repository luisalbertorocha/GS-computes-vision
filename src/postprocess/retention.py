"""Retenção: mantém em disco no máximo `max_images` imagens processadas.

Sem isso a pasta de dados cresce para sempre (cada imagem gera ~0,8 MB entre etapas e
overlay) até o disco encher e o pipeline passar a falhar. Ao passar do limite, saem
primeiro as imagens SEM dano detectado (da mais antiga para a mais recente); imagens com
dano (severidade baixa, média ou alta) só saem quando não sobra nenhuma sem dano. A
imagem removida perde as etapas intermediárias, o overlay e as linhas do parquet.

O que NÃO é removido:
- o id em processed.json, para a imagem nunca ser processada de novo;
- os chamados já abertos (histórico pequeno, já enviado ao webhook);
- fotos colocadas à mão em data/raw e as amostras de demonstração. Só o arquivo bruto
  do Mapillary é apagado, porque é a única fonte que cresce sem teto.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.config import Paths, Settings
from src.ingest.watch_folder import IMAGE_EXTENSIONS, image_id_for
from src.postprocess import output

log = logging.getLogger(__name__)

# pasta que tem exatamente um arquivo por imagem processada (com ou sem dano detectado)
REFERENCE_STEP = "00_original"


def stored_images(paths: Paths) -> list[Path]:
    """Imagens guardadas, da mais antiga para a mais recente."""
    folder = paths.intermediate / REFERENCE_STEP
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.jpg"), key=lambda p: (p.stat().st_mtime, p.name))


def prune_old_images(settings: Settings) -> int:
    """Remove as imagens além de `max_images`: primeiro as sem dano, depois as mais antigas.

    Retorna quantas foram removidas.
    """
    paths = settings.paths
    stored = stored_images(paths)
    excess = len(stored) - settings.max_images
    if settings.max_images <= 0 or excess <= 0:
        return 0
    damaged = {row["imagem_id"] for row in output.read_detections(paths)}
    # ordenação estável: as sem dano vão para a frente, mantendo a ordem de idade
    candidates = sorted(stored, key=lambda p: p.stem in damaged)
    old_ids = {p.stem for p in candidates[:excess]}

    # arquivos brutos do Mapillary dessas imagens (o id é o hash do conteúdo)
    for raw in paths.raw.glob("mapillary_*"):
        if raw.suffix.lower() in IMAGE_EXTENSIONS and image_id_for(raw) in old_ids:
            raw.with_suffix(".json").unlink(missing_ok=True)
            raw.unlink(missing_ok=True)

    for image_id in old_ids:
        (paths.annotated / f"{image_id}.jpg").unlink(missing_ok=True)
        (paths.thumbs / f"{image_id}.jpg").unlink(missing_ok=True)
        for step_dir in paths.intermediate.iterdir():
            if step_dir.is_dir():
                (step_dir / f"{image_id}.jpg").unlink(missing_ok=True)

    rows = output.remove_images(paths, old_ids)
    log.info(
        "Retenção: %d imagem(ns) removida(s), %d delas com dano (%d detecção(ões))",
        len(old_ids), len(old_ids & damaged), rows,
    )
    return len(old_ids)
