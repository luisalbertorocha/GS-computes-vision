"""Pasta monitorada: qualquer imagem nova em data/raw entra no pipeline sozinha."""
from __future__ import annotations

import hashlib
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_id_for(path: Path) -> str:
    """ID estável pelo conteúdo: a mesma foto nunca é processada duas vezes."""
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def find_new_images(raw_dir: Path, processed: set[str]) -> list[tuple[Path, str]]:
    new = []
    seen = set(processed)
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_id = image_id_for(path)
        if image_id not in seen:
            seen.add(image_id)
            new.append((path, image_id))
    return new
