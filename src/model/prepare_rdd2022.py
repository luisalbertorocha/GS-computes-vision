"""Baixa o RDD2022 (Figshare, CC BY 4.0) e converte para o formato YOLO.

    python -m src.model.prepare_rdd2022 --download India Japan United_States Czech

O zip oficial tem 13 GB: um zip por país, guardados sem compressão. Com --download,
só os zips dos países pedidos são lidos de dentro do arquivo remoto (requisições
HTTP Range), sem baixar os 13 GB. Sem --download, converte o que já estiver em --source.

Conversão:
- anotações PASCAL VOC (.xml) -> rótulos YOLO (.txt), caixas normalizadas;
- D00, D10, D20 (trincas longitudinal, transversal e couro de jacaré) -> `trinca`;
  D40 -> `buraco`; as demais classes do dataset (faixas apagadas, tampas, remendos) são ignoradas;
- divisão treino/validação determinística (hash do nome do arquivo), 90/10;
- imagens sem dano entram como fundo, limitadas a uma fração das imagens com dano.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIGSHARE_ZIP = "https://ndownloader.figshare.com/files/38030910"
COUNTRIES = ["India", "Japan", "United_States", "Czech", "Norway", "China_MotorBike", "China_Drone"]

CLASS_NAMES = ["trinca", "buraco"]
LABEL_TO_CLASS = {"D00": 0, "D10": 0, "D20": 0, "D40": 1}


# ---------- download parcial do zip remoto ----------

class HttpRangeFile:
    """Arquivo somente-leitura sobre HTTP Range, com leitura antecipada em blocos grandes."""

    def __init__(self, url: str, chunk: int = 8 << 20):
        self.url, self.chunk = url, chunk
        self.session = requests.Session()
        head = self._get(0, 0)
        self.size = int(head.headers["Content-Range"].split("/")[1])
        self.pos, self.buf, self.buf_start = 0, b"", 0

    def _get(self, start: int, stop: int) -> requests.Response:
        for attempt in range(6):
            try:
                response = self.session.get(
                    self.url, headers={"Range": f"bytes={start}-{stop}"}, timeout=120
                )
                if response.status_code == 206:
                    return response
                response.raise_for_status()
                raise OSError("o servidor não aceitou a requisição parcial (Range)")
            except (requests.RequestException, OSError):
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))
        raise AssertionError("inalcançável")

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, min(self.size, base + offset))
        return self.pos

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        out = bytearray()
        while self.pos < end:
            if not (self.buf_start <= self.pos < self.buf_start + len(self.buf)):
                stop = min(self.pos + max(self.chunk, end - self.pos), self.size) - 1
                self.buf, self.buf_start = self._get(self.pos, stop).content, self.pos
            offset = self.pos - self.buf_start
            piece = self.buf[offset : offset + (end - self.pos)]
            out += piece
            self.pos += len(piece)
        return bytes(out)

    def close(self) -> None:
        self.session.close()


def _download_country(country: str, target: Path) -> None:
    """Baixa `RDD2022/<país>.zip` de dentro do zip remoto e extrai a pasta train."""
    local_zip = target / "_zips" / f"{country}.zip"
    local_zip.parent.mkdir(parents=True, exist_ok=True)
    remote = HttpRangeFile(FIGSHARE_ZIP)
    with zipfile.ZipFile(remote) as archive:
        info = archive.getinfo(f"RDD2022/{country}.zip")
        if not (local_zip.exists() and local_zip.stat().st_size == info.file_size):
            partial = local_zip.with_suffix(".part")
            with archive.open(info) as source, partial.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=8 << 20)
            partial.replace(local_zip)
    remote.close()

    with zipfile.ZipFile(local_zip) as archive:
        members = [n for n in archive.namelist() if f"{country}/train/" in n and not n.endswith("/")]
        for name in members:
            destination = target / name[name.index(f"{country}/train/") :]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
    print(f"  {country}: {local_zip.stat().st_size / 1e6:.0f} MB, {len(members)} arquivos de treino", flush=True)


def download(countries: list[str], target: Path) -> None:
    """Baixa só os países pedidos (um zip por país dentro do zip do Figshare), em paralelo."""
    unknown = [c for c in countries if c not in COUNTRIES]
    if unknown:
        raise SystemExit(f"País desconhecido: {', '.join(unknown)}. Opções: {', '.join(COUNTRIES)}")
    print(f"Baixando {', '.join(countries)} do Figshare...", flush=True)
    with ThreadPoolExecutor(max_workers=len(countries)) as pool:
        list(pool.map(lambda country: _download_country(country, target), countries))


# ---------- conversão VOC -> YOLO ----------

def parse_annotation(xml_path: Path, image_path: Path) -> list[tuple[int, float, float, float, float]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width", "0"))) if size is not None else 0
    height = int(float(size.findtext("height", "0"))) if size is not None else 0
    if width <= 0 or height <= 0:
        with Image.open(image_path) as im:
            width, height = im.size
    boxes = []
    for obj in root.iter("object"):
        cls = LABEL_TO_CLASS.get((obj.findtext("name") or "").strip())
        box = obj.find("bndbox")
        if cls is None or box is None:
            continue
        x0, y0, x1, y1 = (float(box.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax"))
        x0, x1 = sorted((min(max(x0, 0), width), min(max(x1, 0), width)))
        y0, y1 = sorted((min(max(y0, 0), height), min(max(y1, 0), height)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        boxes.append(
            (cls, (x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height)
        )
    return boxes


def _is_val(name: str, val_fraction: float) -> bool:
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 < val_fraction * 1000


def convert(source: Path, out: Path, val_fraction: float = 0.1, background_fraction: float = 0.15) -> dict:
    xml_files = sorted(source.glob("*/train/annotations/xmls/*.xml"))
    if not xml_files:
        raise SystemExit(f"Nenhuma anotação em {source}/<país>/train/annotations/xmls/")

    labeled, background = [], []
    for xml_path in xml_files:
        image_path = xml_path.parents[2] / "images" / f"{xml_path.stem}.jpg"
        if not image_path.exists():
            continue
        boxes = parse_annotation(xml_path, image_path)
        (labeled if boxes else background).append((image_path, boxes))
    # fundos: ordem estável e independente do país
    background.sort(key=lambda item: hashlib.md5(item[0].name.encode()).hexdigest())
    background = background[: int(len(labeled) * background_fraction)]

    if out.exists():
        shutil.rmtree(out)
    stats = {split: {"imagens": 0, "fundos": 0, **{n: 0 for n in CLASS_NAMES}} for split in ("train", "val")}
    countries: Counter = Counter()
    for image_path, boxes in labeled + background:
        split = "val" if _is_val(image_path.name, val_fraction) else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path, out / "images" / split / image_path.name)
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes]
        (out / "labels" / split / f"{image_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        stats[split]["imagens"] += 1
        stats[split]["fundos"] += not boxes
        for c, *_ in boxes:
            stats[split][CLASS_NAMES[c]] += 1
        countries[image_path.parents[2].name] += 1

    # sem `path`: o Ultralytics resolve os caminhos a partir da pasta do data.yaml
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    (out / "data.yaml").write_text(
        f"train: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8"
    )
    summary = {
        "fonte": "RDD2022 (Figshare, doi:10.6084/m9.figshare.21431547.v1)",
        "licenca": "CC BY 4.0",
        "classes": CLASS_NAMES,
        "mapeamento": LABEL_TO_CLASS,
        "imagens_por_pais": dict(countries),
        "splits": stats,
    }
    (out / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara o RDD2022 em formato YOLO")
    parser.add_argument("--download", nargs="+", metavar="PAIS", help=f"países a baixar: {', '.join(COUNTRIES)}")
    parser.add_argument("--source", default=str(ROOT / "data" / "rdd2022"), help="pasta com <país>/train/...")
    parser.add_argument("--out", default=str(ROOT / "data" / "dataset"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--background-fraction", type=float, default=0.15)
    args = parser.parse_args()

    source = Path(args.source)
    if args.download:
        download(args.download, source)
    summary = convert(source, Path(args.out), args.val_fraction, args.background_fraction)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
