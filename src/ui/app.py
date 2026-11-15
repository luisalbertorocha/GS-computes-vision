"""Painel do MVP.   streamlit run src/ui/app.py

Mostra, sem ninguém clicar em "processar": ranking de danos, as transformações
intermediárias de cada imagem, os dados extraídos e o histórico do agendador.
Com RUN_SCHEDULER_IN_UI=1 o próprio painel sobe o agendador em segundo plano
(útil para deploy de processo único, ex.: Hugging Face Spaces).

Atualização: só a faixa de status e as métricas se redesenham a cada 30 s. As abas
levam imagens embutidas (pesadas), então só são redesenhadas quando os dados mudam
ou quando a pessoa interage.
"""
from __future__ import annotations

import base64
import html
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_settings  # noqa: E402
from src.io_utils import read_image  # noqa: E402
from src.postprocess import output  # noqa: E402
from src.postprocess import severity  # noqa: E402
from src.postprocess.geo import TZ_BR  # noqa: E402
from src.postprocess.retention import stored_images  # noqa: E402
from src.preprocess.pipeline import STEP_LABELS  # noqa: E402

st.set_page_config(page_title="Zeladoria Viária · Visão Computacional", page_icon="🛣️", layout="wide")
settings = load_settings()
paths = settings.paths

# Severidade é um estado, não uma série: cores de status fixas, sempre com ícone e texto
# (nunca só a cor). Valores validados para contraste em fundo escuro.
SEVERITY = {
    "alta": ("🔴", "Alta", "#d03b3b"),
    "média": ("🟡", "Média", "#fab219"),
    "baixa": ("🟢", "Baixa", "#0ca30c"),
}
SEVERITY_ORDER = ["alta", "média", "baixa"]

_CSS = """
<style>
.zcv-steps { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem; }
.zcv-steps figure, .zcv-photo { margin: 0; }
.zcv-steps img, .zcv-photo img { width: 100%; height: auto; display: block; border-radius: 0.5rem; }
.zcv-steps figcaption { font-size: 0.85rem; opacity: 0.7; text-align: center; margin-top: 0.35rem; }
.zcv-facts { display: grid; grid-template-columns: minmax(0, 11rem) minmax(0, 1fr); gap: 0.55rem 1.25rem; margin: 0; }
.zcv-facts dt { opacity: 0.65; }
.zcv-facts dd { margin: 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.zcv-sev { display: inline-flex; align-items: center; gap: 0.4rem; font-weight: 600; }
.zcv-sev i { width: 0.7rem; height: 0.7rem; border-radius: 50%; display: inline-block; }
</style>
"""


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


# ---------- formatação ----------

def _br_datetime(iso: str | None, seconds: bool = False) -> str:
    """'2026-11-15T05:18:17-03:00' -> '15/11/2026 05:18' (já vem no fuso de Brasília)."""
    if not iso or len(iso) < 16:
        return "sem data"
    return f"{iso[8:10]}/{iso[5:7]}/{iso[0:4]} {iso[11:19] if seconds else iso[11:16]}"


def _ago(iso: str) -> str:
    """Tempo decorrido em palavras: 'há 20 s', 'há 3 min', 'há 2 h'."""
    seconds = max(0, int((datetime.now(TZ_BR) - datetime.fromisoformat(iso)).total_seconds()))
    if seconds < 60:
        return f"há {seconds} s"
    if seconds < 3600:
        return f"há {seconds // 60} min"
    return f"há {seconds // 3600} h"


def _place(lat, lon) -> str:
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return "—"
    return f"{lat:.5f}, {lon:.5f}"


def _kind(classe: str) -> str:
    return classe.replace("_", " ").capitalize()


def _severity_text(level: str) -> str:
    icon, label, _ = SEVERITY.get(level, ("⚪", level.capitalize(), "#888888"))
    return f"{icon} {label}"


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


# ---------- imagens embutidas ----------

@st.cache_data(max_entries=128, show_spinner=False)
def _data_uri(path: str, mtime: float) -> str:
    """Imagem embutida em base64; `mtime` invalida o cache quando o arquivo muda."""
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


@st.cache_data(max_entries=512, show_spinner=False)
def _thumb_uri(path: str, mtime: float, width: int = 112) -> str | None:
    """Miniatura para a tabela do ranking (poucos KB por linha)."""
    img = read_image(Path(path))
    if img is None:
        return None
    height = max(1, round(img.shape[0] * width / img.shape[1]))
    small = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode() if ok else None


def _embedded(path: Path, thumb: bool = False) -> str | None:
    if not path.exists():
        return None
    return (_thumb_uri if thumb else _data_uri)(str(path), path.stat().st_mtime)


def _steps_gallery(images: list[tuple[str, Path]]) -> str:
    """Grade responsiva com as imagens embutidas: chegam junto com a página, sem download à parte."""
    figures = "".join(
        f'<figure><img src="{_embedded(p)}" alt="{html.escape(label)}">'
        f"<figcaption>{html.escape(label)}</figcaption></figure>"
        for label, p in images
    )
    return f'<div class="zcv-steps">{figures}</div>'


# ---------- nomes legíveis ----------

def _detection_names(df: pd.DataFrame) -> dict[str, str]:
    """Nome curto de cada detecção: 'Imagem 3' ou 'Imagem 3 · dano 2'.

    As imagens são numeradas pela ordem de processamento, então o número de uma imagem
    não muda quando chegam novas. Quando uma imagem tem mais de um dano, eles viram
    'dano 1', 'dano 2'...
    """
    order = df.sort_values(["processed_at", "imagem_id"]).drop_duplicates("imagem_id")["imagem_id"]
    number = {image_id: n for n, image_id in enumerate(order, start=1)}
    per_image = df.groupby("imagem_id")["id"].transform("count")
    names = {}
    for (_, r), count in zip(df.iterrows(), per_image):
        damage = f" · dano {int(r['id'].rsplit('_', 1)[1]) + 1}" if count > 1 else ""
        names[r["id"]] = f"Imagem {number[r['imagem_id']]}{damage}"
    return names


def _detection_labels(df: pd.DataFrame, names: dict[str, str]) -> dict[str, str]:
    """Rótulo do seletor: 'Imagem 3 · buraco (alta) · 15/11/2026 05:18'."""
    return {
        r["id"]: f"{names[r['id']]} · {r['classe']} ({r['severidade']}) · {_br_datetime(r['captured_at'])}"
        for _, r in df.iterrows()
    }


# ---------- faixa de status + métricas (atualiza sozinha) ----------

def _data_version() -> tuple:
    """Muda quando há imagens ou detecções novas (ou removidas); é o gatilho para redesenhar as abas."""
    parquet = paths.parquet.stat().st_mtime_ns if paths.parquet.exists() else 0
    return parquet, len(output.load_state(paths))


@st.fragment(run_every="30s")
def status() -> None:
    version = _data_version()
    if st.session_state.setdefault("_versao", version) != version:
        st.session_state["_versao"] = version
        st.rerun(scope="app")

    rows = output.read_detections(paths)
    runs = output.read_runs(paths, limit=1)
    last = runs[0] if runs else None
    stored = len(stored_images(paths))

    if last and last.get("erros"):
        problems = "\n".join(f"- {error}" for error in last["erros"])
        st.warning(
            f"A última execução ({last['fim'][11:19]}) terminou com "
            f"{_plural(len(last['erros']), 'erro', 'erros')}. O que já foi processado continua no "
            f"painel, e a próxima execução tenta de novo.\n\n{problems}",
            icon="⚠️",
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Danos detectados", len(rows), border=True)
    c2.metric("Severidade alta", sum(r["severidade"] == "alta" for r in rows), border=True)
    c3.metric("Chamados abertos", len(list(paths.tickets.glob("chamado_*.json"))), border=True)
    c4.metric(
        "Imagens guardadas",
        f"{stored} de {settings.max_images}" if settings.max_images > 0 else stored,
        border=True,
        help="Ao passar do limite (MAX_IMAGES), as imagens mais antigas são removidas sozinhas.",
    )
    c5.metric("Última execução", last["fim"][11:19] if last else "—", border=True)

    if last:
        summary = (
            f"Última execução {_ago(last['fim'])}: "
            f"{_plural(last['imagens_coletadas'], 'imagem coletada', 'imagens coletadas')} do Mapillary, "
            f"{_plural(last['imagens_processadas'], 'processada', 'processadas')}, "
            f"{_plural(last['danos_detectados'], 'dano', 'danos')}, "
            f"{_plural(last['chamados_abertos'], 'chamado', 'chamados')}."
        )
    else:
        summary = "O agendador ainda não rodou. A primeira execução começa em instantes."
    first = output.first_run(paths)
    history = f" Histórico desde {_br_datetime(first['inicio'])}." if first else ""
    ephemeral = (
        " Neste servidor de demonstração os dados reiniciam a cada deploy."
        if settings.demo_seed_per_run > 0
        else ""
    )
    st.caption(summary + history + ephemeral)

    def source(active: bool, on: str, off: str) -> str:
        return f":green-badge[:material/check: {on}]" if active else f":gray-badge[{off}]"

    st.markdown(
        " ".join(
            [
                source(
                    bool(settings.mapillary_token and settings.mapillary_bbox),
                    "Mapillary ativo",
                    "Mapillary não configurado",
                ),
                source(settings.demo_seed_per_run > 0, "Modo demonstração ligado", "Modo demonstração desligado"),
                source(bool(settings.webhook_url), "Webhook configurado", "Webhook não configurado"),
            ]
        )
    )


# ---------- abas ----------

def _ranking(tab, df: pd.DataFrame, names: dict[str, str]) -> None:
    with tab:
        levels = st.pills(
            "Severidade",
            SEVERITY_ORDER,
            selection_mode="multi",
            default=SEVERITY_ORDER,
            format_func=_severity_text,
            key="sev",
        )
        shown = df[df["severidade"].isin(levels)]
        if shown.empty:
            st.info("Nenhum dano nas severidades escolhidas. Marque pelo menos uma acima.")
        else:
            table = pd.DataFrame(
                {
                    "Foto": [_embedded(paths.data / p, thumb=True) for p in shown["img_anotada"]],
                    "Imagem": [names[i] for i in shown["id"]],
                    "Tipo": shown["classe"].map(_kind).tolist(),
                    "Severidade": shown["severidade"].map(_severity_text).tolist(),
                    "Score": shown["score_severidade"].tolist(),
                    "Confiança": (shown["confianca"] * 100).tolist(),
                    "Área": shown["area_cm2"].tolist(),
                    "Local": [_place(lat, lon) for lat, lon in zip(shown["lat"], shown["lon"])],
                    "Capturada em": shown["captured_at"].map(_br_datetime).tolist(),
                    "Fonte": shown["fonte"].tolist(),
                }
            )
            st.dataframe(
                table,
                hide_index=True,
                use_container_width=True,
                row_height=64,
                height=min(680, 40 + 64 * len(table)),  # até 10 linhas visíveis; depois, rolagem
                column_config={
                    "Foto": st.column_config.ImageColumn("Foto", width="small"),
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0.0, max_value=1.0, format="%.2f",
                        help="Prioridade de 0 a 1 (veja a aba Como o score funciona).",
                    ),
                    "Confiança": st.column_config.NumberColumn("Confiança", format="%.0f%%"),
                    "Área": st.column_config.NumberColumn("Área", format="%.0f cm²"),
                },
            )
        st.download_button(
            "Baixar detections.parquet", paths.parquet.read_bytes(), "detections.parquet"
        )


def _facts(row: dict, name: str) -> str:
    _, label, color = SEVERITY.get(row["severidade"], ("", row["severidade"], "#888888"))
    facts = [
        ("Detecção", html.escape(name)),
        ("Tipo", html.escape(_kind(row["classe"]))),
        ("Severidade", f'<span class="zcv-sev"><i style="background:{color}"></i>{html.escape(label)}</span>'),
        ("Score", f"{row['score_severidade']:.2f} de 1"),
        ("Confiança", f"{row['confianca'] * 100:.0f}%"),
        ("Área estimada", f"{row['area_cm2']:,.0f} cm²".replace(",", ".")),
        ("Local", _place(row["lat"], row["lon"])),
        ("Capturada em", _br_datetime(row["captured_at"])),
        ("Processada em", _br_datetime(row["processed_at"])),
        ("Fonte", html.escape(str(row["fonte"]))),
        ("Modelo", html.escape(str(row["modelo"]))),
        ("Rostos e placas desfocados", str(row["regioes_anonimizadas"])),
    ]
    return '<dl class="zcv-facts">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in facts) + "</dl>"


def _transformations(tab, rows: list[dict], df: pd.DataFrame, names: dict[str, str]) -> None:
    with tab:
        ids = df["id"].tolist()
        if st.session_state.get("det") not in ids:  # a detecção escolhida pode ter sido removida
            st.session_state.pop("det", None)
        labels = _detection_labels(df, names)
        chosen = st.selectbox("Detecção", ids, key="det", format_func=labels.get)
        row = next(r for r in rows if r["id"] == chosen)
        steps = [
            (label, paths.intermediate / step / f"{row['imagem_id']}.jpg")
            for step, label in STEP_LABELS.items()
        ]
        steps = [(label, p) for label, p in steps if p.exists()]
        annotated = _embedded(paths.data / row["img_anotada"])

        # Aviso de carregamento só quando a detecção muda. Não usa st.spinner: ele só aparece
        # após 0,5 s, e ler os arquivos é mais rápido que isso; a espera real é a transferência
        # das imagens até o navegador. Como as imagens vão embutidas no HTML, o aviso só some
        # depois que elas chegam.
        changed = st.session_state.get("_det_exibida") != chosen
        loading = st.empty()
        if changed:
            loading.info("⏳ Carregando as transformações…")

        photo, details = st.columns([3, 2], gap="large")
        if annotated:
            photo.html(
                f'<figure class="zcv-photo"><img src="{annotated}" '
                f'alt="Foto com os danos detectados marcados"></figure>'
            )
        details.html(_facts(row, names[chosen]))

        st.subheader("Etapas do pré-processamento")
        st.html(_steps_gallery(steps))
        loading.empty()
        st.session_state["_det_exibida"] = chosen

        with st.expander("Dados brutos (JSON)"):
            st.json(row)


def _map(tab, df: pd.DataFrame) -> None:
    with tab:
        geo = df.dropna(subset=["lat", "lon"]).copy()
        if geo.empty:
            st.info(
                "Nenhuma detecção com localização ainda. As amostras de demonstração não têm "
                "coordenadas; fotos do Mapillary ou com GPS aparecem aqui."
            )
            return
        geo["cor"] = geo["severidade"].map(lambda s: SEVERITY.get(s, ("", "", "#888888"))[2])
        geo["tamanho"] = 12 + 40 * geo["score_severidade"]  # raio em metros
        st.map(geo, latitude="lat", longitude="lon", color="cor", size="tamanho")
        st.caption(
            " ".join(_severity_text(level) for level in SEVERITY_ORDER)
            + ". Quanto maior o ponto, maior o score. "
            + _plural(len(geo), "dano com localização", "danos com localização")
            + f" de {len(df)}."
        )


def _tickets(tab, names: dict[str, str]) -> None:
    with tab:
        if settings.webhook_url:
            st.caption("Webhook configurado: cada chamado novo é enviado na hora em que é aberto.")
        else:
            st.caption(
                "Webhook não configurado: os chamados ficam só nos arquivos JSON em "
                "`data/out/chamados/`. Defina `WEBHOOK_URL` para enviá-los a um robô."
            )
        tickets = output.read_tickets(paths)
        if not tickets:
            st.info(
                "Nenhum chamado ainda. Um chamado é aberto quando um dano de severidade alta "
                f"tem confiança de pelo menos {settings.ticket_min_confidence:.0%}."
            )
            return
        table = pd.DataFrame(
            {
                "Detecção": [names.get(t["id"], "Imagem já removida") for t in tickets],
                "Tipo": [_kind(t["classe"]) for t in tickets],
                "Score": [t["score_severidade"] for t in tickets],
                "Área": [t["area_cm2"] for t in tickets],
                "Local": [_place(t.get("lat"), t.get("lon")) for t in tickets],
                "Descrição enviada": [t["descricao"] for t in tickets],
            }
        )
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.2f"),
                "Área": st.column_config.NumberColumn("Área", format="%.0f cm²"),
                "Descrição enviada": st.column_config.TextColumn("Descrição enviada", width="large"),
            },
        )


@st.fragment(run_every="30s")
def _runs_table() -> None:
    """Fragmento próprio: a lista de execuções muda a cada minuto e é leve, então se atualiza sozinha."""
    runs = output.read_runs(paths, limit=30)
    st.caption("Cada linha é uma execução do agendador, sem intervenção humana.")
    if not runs:
        st.info("O agendador ainda não rodou. A primeira execução começa em instantes.")
        return

    def duration(run: dict) -> float:
        return (datetime.fromisoformat(run["fim"]) - datetime.fromisoformat(run["inicio"])).total_seconds()

    table = pd.DataFrame(
        {
            "Resultado": ["⚠️ Com erro" if r.get("erros") else "✅ Concluída" for r in runs],
            "Início": [_br_datetime(r["inicio"], seconds=True) for r in runs],
            "Duração": [duration(r) for r in runs],
            "Coletadas": [r["imagens_coletadas"] for r in runs],
            "Amostras": [r["imagens_amostra"] for r in runs],
            "Processadas": [r["imagens_processadas"] for r in runs],
            "Danos": [r["danos_detectados"] for r in runs],
            "Chamados": [r["chamados_abertos"] for r in runs],
            "Removidas": [r.get("imagens_removidas", 0) for r in runs],
            "Erros": ["; ".join(r.get("erros") or []) for r in runs],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Duração": st.column_config.NumberColumn("Duração", format="%.0f s"),
            "Coletadas": st.column_config.NumberColumn("Coletadas", help="Imagens novas baixadas do Mapillary."),
            "Amostras": st.column_config.NumberColumn("Amostras", help="Imagens de exemplo do modo demonstração."),
            "Removidas": st.column_config.NumberColumn(
                "Removidas", help="Imagens antigas apagadas por passarem do limite de imagens guardadas."
            ),
            "Erros": st.column_config.TextColumn("Erros", width="large"),
        },
    )


def _about(tab) -> None:
    with tab:
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


@st.fragment
def tabs() -> None:
    rows = output.read_detections(paths)
    tab_rank, tab_steps, tab_map, tab_tickets, tab_runs, tab_about = st.tabs(
        ["Ranking de prioridade", "Transformações", "Mapa", "Chamados", "Execuções automáticas",
         "Como o score funciona"]
    )
    names: dict[str, str] = {}
    if not rows:
        empty = (
            "Nenhum dano registrado ainda. Coloque imagens em `data/raw`, configure o Mapillary "
            "ou ligue o modo demonstração; elas entram na próxima execução do agendador."
        )
        for tab in (tab_rank, tab_steps, tab_map):
            tab.info(empty)
    else:
        df = pd.DataFrame(rows).sort_values("score_severidade", ascending=False)
        names = _detection_names(df)
        _ranking(tab_rank, df, names)
        _transformations(tab_steps, rows, df, names)
        _map(tab_map, df)
    _tickets(tab_tickets, names)
    with tab_runs:
        _runs_table()
    _about(tab_about)


if os.getenv("RUN_SCHEDULER_IN_UI") == "1":
    _start_scheduler()

st.html(_CSS)
st.title("🛣️ Zeladoria Viária com Visão Computacional")
st.caption(
    "Imagens de rua entram sozinhas, passam por anonimização e pré-processamento, "
    "e cada dano detectado é medido, priorizado e vira um chamado."
)
status()
tabs()
