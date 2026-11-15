"""Painel do MVP.   streamlit run src/ui/app.py

Mostra, sem ninguém clicar em "processar": ranking de danos, as transformações
intermediárias de cada imagem, os dados extraídos e o histórico do agendador.
Com RUN_SCHEDULER_IN_UI=1 o próprio painel sobe o agendador em segundo plano
(útil para deploy de processo único, ex.: Hugging Face Spaces).
"""
from __future__ import annotations

import base64
import html
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_settings  # noqa: E402
from src.postprocess import output  # noqa: E402
from src.postprocess import severity  # noqa: E402
from src.preprocess.pipeline import STEP_LABELS  # noqa: E402

st.set_page_config(page_title="Zeladoria Viária · Visão Computacional", page_icon="🛣️", layout="wide")
settings = load_settings()
paths = settings.paths


@st.cache_resource
def _start_scheduler() -> bool:
    """Sobe o agendador uma única vez por processo."""
    from src.model.infer import load_detector
    from src.pipeline import run_once

    def loop() -> None:
        detector = load_detector(settings.weights, settings.img_size, settings.conf_threshold)
        while True:
            try:
                run_once(settings, detector)
            except Exception as exc:  # o painel continua no ar mesmo se uma execução falhar
                print(f"[agendador] falha: {exc}")
            time.sleep(settings.interval_seconds)

    threading.Thread(target=loop, daemon=True, name="agendador").start()
    return True


if os.getenv("RUN_SCHEDULER_IN_UI") == "1":
    _start_scheduler()

st.title("🛣️ Zeladoria Viária com Visão Computacional")
st.caption(
    "Imagens de rua entram sozinhas, passam por anonimização e pré-processamento, "
    "e cada dano detectado é medido, priorizado e vira um chamado."
)


@st.cache_data(max_entries=128, show_spinner=False)
def _data_uri(path: str, mtime: float) -> str:
    """Imagem embutida em base64; `mtime` invalida o cache quando o arquivo muda."""
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


_GALLERY_CSS = """
<style>
.zcv-steps { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem; }
.zcv-steps figure { margin: 0; }
.zcv-steps img { width: 100%; height: auto; display: block; border-radius: 0.5rem; }
.zcv-steps figcaption { font-size: 0.85rem; opacity: 0.7; text-align: center; margin-top: 0.35rem; }
</style>
"""


def _steps_gallery(images: list[tuple[str, Path]]) -> str:
    """Grade responsiva com as imagens embutidas: chegam junto com a página, sem download à parte."""
    figures = "".join(
        f'<figure><img src="{_data_uri(str(p), p.stat().st_mtime)}" alt="{html.escape(label)}">'
        f"<figcaption>{html.escape(label)}</figcaption></figure>"
        for label, p in images
    )
    return f'{_GALLERY_CSS}<div class="zcv-steps">{figures}</div>'


@st.fragment(run_every="30s")
def dashboard() -> None:
    rows = output.read_detections(paths)
    runs = output.read_runs(paths, limit=30)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Danos detectados", len(rows))
    c2.metric("Severidade alta", sum(r["severidade"] == "alta" for r in rows))
    c3.metric("Imagens processadas", len(output.load_state(paths)))
    c4.metric("Última execução", runs[0]["fim"][11:19] if runs else "—")

    tab_rank, tab_steps, tab_map, tab_tickets, tab_runs, tab_about = st.tabs(
        ["Ranking de prioridade", "Transformações", "Mapa", "Chamados", "Execuções automáticas",
         "Como o score funciona"]
    )

    with tab_tickets:
        tickets = output.read_tickets(paths)
        st.write(
            "Todo dano de severidade alta vira um chamado: um arquivo JSON em `data/out/chamados/` "
            "e um POST no webhook do robô RPA, quando configurado."
        )
        if tickets:
            st.dataframe(
                pd.DataFrame(tickets)[["id", "classe", "score_severidade", "area_cm2", "lat", "lon", "descricao"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Nenhum chamado aberto ainda.")

    if not rows:
        tab_rank.info("Nenhum dano registrado ainda. Coloque imagens em data/raw ou configure o Mapillary.")
    else:
        df = pd.DataFrame(rows).sort_values("score_severidade", ascending=False)
        levels = tab_rank.multiselect(
            "Severidade", ["alta", "média", "baixa"], default=["alta", "média", "baixa"], key="sev"
        )
        tab_rank.dataframe(
            df[df["severidade"].isin(levels)][
                ["id", "classe", "severidade", "score_severidade", "confianca", "area_cm2",
                 "lat", "lon", "captured_at", "fonte", "modelo"]
            ],
            hide_index=True,
            use_container_width=True,
        )
        tab_rank.download_button(
            "Baixar detections.parquet", paths.parquet.read_bytes(), "detections.parquet"
        )

        with tab_steps:
            ids = df["id"].tolist()
            chosen = st.selectbox("Detecção", ids, key="det")
            row = next(r for r in rows if r["id"] == chosen)
            images = [
                (label, paths.intermediate / step / f"{row['imagem_id']}.jpg")
                for step, label in STEP_LABELS.items()
            ]
            images.append(("Detecções anotadas", paths.data / row["img_anotada"]))
            images = [(label, p) for label, p in images if p.exists()]
            # Aviso de carregamento só quando a detecção muda (não a cada atualização de 30 s).
            # Não usa st.spinner: ele só aparece após 0,5 s, e ler os arquivos é mais rápido
            # que isso; a espera real é a transferência das imagens até o navegador. Como as
            # imagens vão embutidas no HTML, o aviso só some depois que elas chegam.
            changed = st.session_state.get("_det_exibida") != chosen
            loading = st.empty()
            if changed:
                loading.info("⏳ Carregando as transformações…")
            st.html(_steps_gallery(images))
            loading.empty()
            st.session_state["_det_exibida"] = chosen
            st.subheader("Dados extraídos")
            st.json(row)

        with tab_map:
            geo = df.dropna(subset=["lat", "lon"])
            if geo.empty:
                st.info("Nenhuma detecção com coordenadas (fotos sem GPS no EXIF nem sidecar).")
            else:
                st.map(geo, latitude="lat", longitude="lon")

    with tab_runs:
        st.write("Cada linha é uma execução do agendador, sem intervenção humana.")
        if runs:
            st.dataframe(pd.DataFrame(runs), hide_index=True, use_container_width=True)
        else:
            st.info("O agendador ainda não rodou.")

    with tab_about:
        st.markdown(
            f"""
**score = {severity.W_AREA} × área normalizada + {severity.W_CONF} × confiança + {severity.W_CLASSE} × peso da classe**

- área normalizada = min(área em cm² / {severity.AREA_REF_CM2:.0f}, 1)
- peso da classe: {", ".join(f"{k} = {v}" for k, v in severity.CLASS_WEIGHTS.items())}
- faixas: baixa < {severity.THRESHOLD_MEDIA} ≤ média < {severity.THRESHOLD_ALTA} ≤ alta

A área é uma estimativa (escala fixa de pixels para centímetros). Casos de severidade
**alta** geram um chamado e devem ser validados por uma pessoa antes do reparo.
"""
        )


dashboard()
