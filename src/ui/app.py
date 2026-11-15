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
from src.pipeline import draw_overlay  # noqa: E402
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
NO_DAMAGE_COLOR = "#b4bac4"  # imagens analisadas sem dano (mapa)
IMAGE_PREFIX = "img:"  # no seletor, separa "imagem sem dano" de um id de detecção

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
/* o seletor é um campo com busca; sem isto o cursor é o de texto e não parece clicável */
[data-testid="stSelectbox"] input { cursor: pointer; }
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

@st.cache_data(max_entries=48, show_spinner=False)
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


def _row_thumb(image_id: str, annotated: Path) -> str | None:
    """Miniatura gravada pelo pipeline; dados antigos, sem ela, reduzem a foto anotada na hora."""
    return _embedded(paths.thumbs / f"{image_id}.jpg") or _embedded(annotated, thumb=True)


@st.cache_data(max_entries=32, show_spinner=False)
def _highlight_uri(original: str, mtime: float, chosen: str, boxes: tuple) -> str | None:
    """Foto com o dano escolhido em destaque e os outros danos da imagem em cinza."""
    img = read_image(Path(original))
    if img is None:
        return None
    records = [
        {"id": i, "bbox": list(bbox), "classe": classe, "confianca": conf, "severidade": sev}
        for i, bbox, classe, conf, sev in boxes
    ]
    overlay = draw_overlay(img, records, highlight_id=chosen)
    ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode() if ok else None


def _steps_gallery(images: list[tuple[str, Path]]) -> str:
    """Grade responsiva com as imagens embutidas: chegam junto com a página, sem download à parte."""
    figures = "".join(
        f'<figure><img src="{_embedded(p)}" alt="{html.escape(label)}">'
        f"<figcaption>{html.escape(label)}</figcaption></figure>"
        for label, p in images
    )
    return f'<div class="zcv-steps">{figures}</div>'


# ---------- nomes legíveis ----------

def _image_numbers(images: list[dict], rows: list[dict]) -> dict[str, int]:
    """Número de cada imagem pela ordem de processamento: não muda quando chegam novas.

    Dados antigos, sem images.parquet, são numerados só pelas imagens que têm detecção.
    """
    known = [im["imagem_id"] for im in images]  # já vem na ordem de processamento
    extra = sorted({(r["processed_at"], r["imagem_id"]) for r in rows if r["imagem_id"] not in set(known)})
    return {image_id: n for n, image_id in enumerate(known + [image_id for _, image_id in extra], start=1)}


def _detection_names(df: pd.DataFrame, numbers: dict[str, int]) -> dict[str, str]:
    """Nome curto de cada detecção: 'Imagem 3' ou, com mais de um dano na imagem, 'Imagem 3 · dano 2'."""
    per_image = df.groupby("imagem_id")["id"].transform("count")
    names = {}
    for (_, r), count in zip(df.iterrows(), per_image):
        damage = f" · dano {int(r['id'].rsplit('_', 1)[1]) + 1}" if count > 1 else ""
        names[r["id"]] = f"Imagem {numbers[r['imagem_id']]}{damage}"
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
    clean = sum(im["danos"] == 0 for im in output.read_images(paths))
    no_damage = (
        f" {_plural(clean, 'imagem analisada não teve', 'imagens analisadas não tiveram')} dano detectado."
        if clean
        else ""
    )
    st.caption(summary + no_damage + history + ephemeral)

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
                    "Foto": [_row_thumb(i, paths.data / p) for i, p in zip(shown["imagem_id"], shown["img_anotada"])],
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


def _facts_html(facts: list[tuple[str, str]]) -> str:
    return '<dl class="zcv-facts">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in facts) + "</dl>"


def _facts(row: dict, name: str) -> str:
    _, label, color = SEVERITY.get(row["severidade"], ("", row["severidade"], "#888888"))
    return _facts_html(
        [
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
    )


def _image_facts(image: dict, name: str) -> str:
    return _facts_html(
        [
            ("Imagem", html.escape(name)),
            ("Resultado", "Nenhum dano detectado"),
            ("Local", _place(image["lat"], image["lon"])),
            ("Capturada em", _br_datetime(image["captured_at"])),
            ("Processada em", _br_datetime(image["processed_at"])),
            ("Fonte", html.escape(str(image["fonte"]))),
            ("Rostos e placas desfocados", str(image["regioes_anonimizadas"])),
        ]
    )


def _damage_photo(row: dict, rows: list[dict]) -> str | None:
    """Foto com o dano escolhido em destaque; sem a original guardada, cai na foto anotada."""
    original = paths.intermediate / "00_original" / f"{row['imagem_id']}.jpg"
    if original.exists():
        boxes = tuple(
            (r["id"], tuple(r["bbox"]), r["classe"], r["confianca"], r["severidade"])
            for r in rows
            if r["imagem_id"] == row["imagem_id"]
        )
        highlighted = _highlight_uri(str(original), original.stat().st_mtime, row["id"], boxes)
        if highlighted:
            return highlighted
    return _embedded(paths.data / row["img_anotada"])


def _transformations(
    tab, rows: list[dict], df: pd.DataFrame | None, clean: list[dict], numbers: dict[str, int], names: dict[str, str]
) -> None:
    """Seletor com todas as imagens analisadas: primeiro os danos (por score), depois as sem dano."""
    with tab:
        labels = _detection_labels(df, names) if df is not None else {}
        ids = df["id"].tolist() if df is not None else []
        clean_by_key = {IMAGE_PREFIX + im["imagem_id"]: im for im in reversed(clean)}  # mais recentes primeiro
        for key, im in clean_by_key.items():
            labels[key] = f"Imagem {numbers[im['imagem_id']]} · sem dano · {_br_datetime(im['captured_at'])}"
        ids += list(clean_by_key)
        if st.session_state.get("det") not in ids:  # a escolhida pode ter sido removida pelo limite
            st.session_state.pop("det", None)
        chosen = st.selectbox(
            "Imagem analisada",
            ids,
            key="det",
            format_func=labels.get,
            help="Primeiro os danos, do score mais alto para o mais baixo; depois as imagens sem dano.",
        )

        row = None
        if chosen in clean_by_key:
            image = clean_by_key[chosen]
            image_id = image["imagem_id"]
            photo_uri = _embedded(paths.annotated / f"{image_id}.jpg")
            alt = "Foto analisada, sem danos detectados"
            facts = _image_facts(image, f"Imagem {numbers[image_id]}")
        else:
            row = next(r for r in rows if r["id"] == chosen)
            image_id = row["imagem_id"]
            photo_uri = _damage_photo(row, rows)
            alt = "Foto com o dano escolhido em destaque"
            facts = _facts(row, names[chosen])
        steps = [
            (label, paths.intermediate / step / f"{image_id}.jpg") for step, label in STEP_LABELS.items()
        ]
        steps = [(label, p) for label, p in steps if p.exists()]

        # Aviso de carregamento só quando a escolha muda. Não usa st.spinner: ele só aparece
        # após 0,5 s, e ler os arquivos é mais rápido que isso; a espera real é a transferência
        # das imagens até o navegador. Como as imagens vão embutidas no HTML, o aviso só some
        # depois que elas chegam.
        changed = st.session_state.get("_det_exibida") != chosen
        loading = st.empty()
        if changed:
            loading.info("⏳ Carregando as transformações…")

        photo, details = st.columns([3, 2], gap="large")
        if photo_uri:
            photo.html(f'<figure class="zcv-photo"><img src="{photo_uri}" alt="{alt}"></figure>')
        details.html(facts)

        st.subheader("Etapas do pré-processamento")
        st.html(_steps_gallery(steps))
        loading.empty()
        st.session_state["_det_exibida"] = chosen

        if row is not None:
            with st.expander("Dados brutos (JSON)"):
                st.json(row)


def _rgb(hex_color: str, alpha: int = 230) -> list[int]:
    return [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)] + [alpha]


def _map(tab, df: pd.DataFrame | None, clean: list[dict], numbers: dict[str, int], names: dict[str, str]) -> None:
    """Danos coloridos por severidade e, em cinza, as imagens analisadas sem dano (área coberta)."""
    import pydeck as pdk

    with tab:
        points = [
            {
                "lat": im["lat"],
                "lon": im["lon"],
                "cor": _rgb(NO_DAMAGE_COLOR, 215),
                "raio": 5,
                "titulo": f"Imagem {numbers[im['imagem_id']]}",
                "detalhe": "Sem dano detectado",
            }
            for im in clean
            if im["lat"] is not None and im["lon"] is not None
        ]
        covered = len(points)
        damages = 0
        if df is not None:
            # maiores primeiro: quando vários danos caem no mesmo ponto, os menores ficam por cima
            for _, r in df.dropna(subset=["lat", "lon"]).iterrows():
                _, label, color = SEVERITY.get(r["severidade"], ("", r["severidade"], "#888888"))
                points.append(
                    {
                        "lat": r["lat"],
                        "lon": r["lon"],
                        "cor": _rgb(color),
                        "raio": 7 + 9 * r["score_severidade"],
                        "titulo": names[r["id"]],
                        "detalhe": (
                            f"{_kind(r['classe'])}, severidade {label.lower()}, "
                            f"score {r['score_severidade']:.2f}"
                        ),
                    }
                )
                damages += 1
        if not points:
            st.info(
                "Nenhuma imagem com localização ainda. As amostras de demonstração não têm "
                "coordenadas; fotos do Mapillary ou com GPS aparecem aqui assim que são analisadas."
            )
            return
        data = pd.DataFrame(points)
        view = pdk.data_utils.compute_view(data[["lon", "lat"]].values.tolist(), view_proportion=1)
        view.zoom = min(view.zoom, 16)  # um ponto só não deve abrir no zoom máximo
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=data,
            get_position="[lon, lat]",
            get_fill_color="cor",
            get_radius="raio",
            # tamanho fixo na tela: o ponto não some ao afastar o mapa. Precisa de types.String:
            # o pydeck trata texto puro como expressão, e o raio viraria gigante.
            radius_units=pdk.types.String("pixels"),
            stroked=True,
            get_line_color=[27, 30, 35],
            line_width_min_pixels=1,
            pickable=True,
        )
        st.pydeck_chart(
            pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"html": "<b>{titulo}</b><br/>{detalhe}"}),
            height=520,
        )
        st.caption(
            " ".join(_severity_text(level) for level in SEVERITY_ORDER)
            + " ⚪ Sem dano. Quanto maior o ponto, maior o score. Passe o mouse sobre um ponto para ver o que é. "
            + f"No mapa: {_plural(damages, 'dano', 'danos')} e "
            + f"{_plural(covered, 'imagem sem dano', 'imagens sem dano')}."
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
    images = output.read_images(paths)
    clean = [im for im in images if im["danos"] == 0]
    numbers = _image_numbers(images, rows)
    tab_rank, tab_steps, tab_map, tab_tickets, tab_runs, tab_about = st.tabs(
        ["Ranking de prioridade", "Transformações", "Mapa", "Chamados", "Execuções automáticas",
         "Como o score funciona"]
    )
    df = pd.DataFrame(rows).sort_values("score_severidade", ascending=False) if rows else None
    names = _detection_names(df, numbers) if df is not None else {}

    if df is not None:
        _ranking(tab_rank, df, names)
    elif clean:
        tab_rank.info(
            f"Nenhum dano detectado nas {_plural(len(clean), 'imagem analisada', 'imagens analisadas')} "
            "até agora. Veja cada uma na aba Transformações."
        )
    if df is None and not clean:
        empty = (
            "Nenhuma imagem analisada ainda. Coloque imagens em `data/raw`, configure o Mapillary "
            "ou ligue o modo demonstração; elas entram na próxima execução do agendador."
        )
        for tab in (tab_rank, tab_steps, tab_map):
            tab.info(empty)
    else:
        _transformations(tab_steps, rows, df, clean, numbers, names)
        _map(tab_map, df, clean, numbers, names)
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
