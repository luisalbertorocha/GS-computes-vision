"""Configuração central: caminhos e parâmetros lidos de variáveis de ambiente (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    data: Path

    @property
    def raw(self) -> Path:  # imagens brutas (nunca publicadas nem versionadas)
        return self.data / "raw"

    @property
    def intermediate(self) -> Path:  # uma subpasta por etapa
        return self.data / "intermediate"

    @property
    def out(self) -> Path:
        return self.data / "out"

    @property
    def annotated(self) -> Path:
        return self.out / "annotated"

    @property
    def tickets(self) -> Path:  # gatilho por arquivo para o robô RPA
        return self.out / "chamados"

    @property
    def logs(self) -> Path:
        return self.out / "logs"

    @property
    def parquet(self) -> Path:
        return self.out / "detections.parquet"

    @property
    def state(self) -> Path:  # hashes das imagens já processadas
        return self.out / "processed.json"

    @property
    def runs_log(self) -> Path:  # uma linha JSON por execução (evidência da automação)
        return self.out / "runs.jsonl"

    @property
    def mapillary_seen(self) -> Path:  # ids do Mapillary já baixados (mesmo que a foto já tenha sido removida)
        return self.out / "mapillary_seen.json"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.intermediate, self.annotated, self.tickets, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def rel(self, path: Path) -> str:
        """Caminho relativo à pasta de dados, sempre com '/', para gravar no parquet."""
        return path.relative_to(self.data).as_posix()


@dataclass(frozen=True)
class Settings:
    paths: Paths
    img_size: int = 640
    conf_threshold: float = 0.25
    weights: Path | None = None
    # Largura real (cm) que a imagem cobre no chão; converte pixels em cm².
    scene_width_cm: float = 400.0
    # "letterbox" = detector recebe a imagem só redimensionada (como no treino);
    # "processed" = após CLAHE + bilateral. Medido em compare_baseline: letterbox tem recall maior.
    detect_on: str = "letterbox"
    webhook_url: str | None = None
    # confiança mínima para um dano de severidade alta abrir chamado automaticamente
    ticket_min_confidence: float = 0.4
    mapillary_token: str | None = None
    mapillary_bbox: str | None = None  # "minLon,minLat,maxLon,maxLat"
    mapillary_limit: int = 10
    default_source: str = "fotos_proprias"
    default_license: str = "proprio"
    interval_seconds: int = 300
    # modo demonstração: quantas imagens de samples/ entram por execução (0 = desligado)
    demo_seed_per_run: int = 0
    samples_dir: Path = ROOT / "samples"
    # quantas imagens processadas ficam guardadas; ao passar disso as mais antigas são
    # removidas (0 = sem limite)
    max_images: int = 200


def _env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def load_settings(data_dir: Path | str | None = None) -> Settings:
    data = Path(data_dir or _env("DATA_DIR") or ROOT / "data")
    weights = _env("MODEL_WEIGHTS")
    if weights is None:
        for candidate in (ROOT / "weights" / "best.onnx", ROOT / "weights" / "best.pt"):
            if candidate.exists():
                weights = str(candidate)
                break
    return Settings(
        paths=Paths(data).ensure(),
        img_size=int(_env("IMG_SIZE") or 640),
        conf_threshold=float(_env("CONF_THRESHOLD") or 0.25),
        weights=Path(weights) if weights else None,
        scene_width_cm=float(_env("SCENE_WIDTH_CM") or 400.0),
        detect_on=_env("DETECT_ON") or "letterbox",
        webhook_url=_env("WEBHOOK_URL"),
        ticket_min_confidence=float(_env("TICKET_MIN_CONFIDENCE") or 0.4),
        mapillary_token=_env("MAPILLARY_TOKEN"),
        mapillary_bbox=_env("MAPILLARY_BBOX"),
        mapillary_limit=int(_env("MAPILLARY_LIMIT") or 10),
        default_source=_env("DEFAULT_SOURCE") or "fotos_proprias",
        default_license=_env("DEFAULT_LICENSE") or "proprio",
        interval_seconds=int(_env("PIPELINE_INTERVAL_SECONDS") or 300),
        demo_seed_per_run=int(_env("DEMO_SEED_PER_RUN") or 0),
        samples_dir=Path(_env("SAMPLES_DIR") or ROOT / "samples"),
        max_images=int(_env("MAX_IMAGES") or 200),
    )
