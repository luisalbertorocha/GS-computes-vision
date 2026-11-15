import json

import cv2
import numpy as np

from src.model.compare_baseline import iou, match
from src.model.infer import Detection
from src.model.prepare_rdd2022 import convert

VOC = """<annotation>
  <size><width>{w}</width><height>{h}</height></size>
  {objects}
</annotation>"""
OBJ = "<object><name>{name}</name><bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox></object>"


def _write(country_dir, stem, objects, size=(200, 100)):
    (country_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (country_dir / "train" / "annotations" / "xmls").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(country_dir / "train" / "images" / f"{stem}.jpg"), np.zeros((size[1], size[0], 3), np.uint8))
    xml = VOC.format(w=size[0], h=size[1], objects="".join(OBJ.format(**o) for o in objects))
    (country_dir / "train" / "annotations" / "xmls" / f"{stem}.xml").write_text(xml, encoding="utf-8")


def test_convert_rdd2022_to_yolo(tmp_path):
    source, out = tmp_path / "rdd", tmp_path / "dataset"
    for i in range(20):
        _write(
            source / "India",
            f"India_{i:06d}",
            [
                dict(name="D40", x0=50, y0=20, x1=150, y1=60),  # buraco
                dict(name="D00", x0=0, y0=0, x1=20, y1=100),  # trinca
                dict(name="D44", x0=0, y0=0, x1=50, y1=50),  # classe ignorada
                dict(name="D40", x0=10, y0=10, x1=10, y1=40),  # caixa degenerada
            ],
        )
    for i in range(10):
        _write(source / "Japan", f"Japan_{i:06d}", [])  # sem dano -> fundo

    summary = convert(source, out, val_fraction=0.2, background_fraction=0.1)

    splits = summary["splits"]
    assert splits["train"]["imagens"] + splits["val"]["imagens"] == 22  # 20 com dano + 2 fundos (10%)
    assert splits["train"]["fundos"] + splits["val"]["fundos"] == 2
    assert splits["train"]["buraco"] + splits["val"]["buraco"] == 20
    assert splits["train"]["trinca"] + splits["val"]["trinca"] == 20
    assert splits["val"]["imagens"] >= 1 and summary["imagens_por_pais"]["India"] == 20

    label = next((out / "labels").rglob("India_*.txt")).read_text(encoding="utf-8").splitlines()
    assert label == ["1 0.500000 0.400000 0.500000 0.400000", "0 0.050000 0.500000 0.100000 1.000000"]
    assert (out / "data.yaml").read_text(encoding="utf-8").splitlines()[-2:] == ["  0: trinca", "  1: buraco"]
    assert json.loads((out / "stats.json").read_text(encoding="utf-8"))["licenca"] == "CC BY 4.0"
    # a mesma entrada gera sempre a mesma divisão
    assert convert(source, out, val_fraction=0.2, background_fraction=0.1) == summary


def test_iou_and_matching():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 10, 10)) - 1 / 3) < 1e-9

    truth = [("buraco", (0, 0, 100, 100)), ("trinca", (200, 0, 20, 100))]
    predictions = [
        Detection("buraco", 0.9, (5, 5, 100, 100), 1e4),  # acerto
        Detection("buraco", 0.8, (0, 0, 100, 100), 1e4),  # duplicata -> falso positivo
        Detection("buraco", 0.7, (200, 0, 20, 100), 2e3),  # classe errada -> falso positivo
    ]
    assert match(predictions, truth) == (1, 2, 1)
    assert match([], truth) == (0, 0, 2)
