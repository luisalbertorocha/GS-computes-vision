import re

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.pipeline import run_once
from src.postprocess import output


def test_api_smoke(settings, raw_image):
    run_once(settings)
    client = TestClient(create_app(settings))

    assert client.get("/health").json()["status"] == "ok"

    rows = client.get("/detections").json()
    assert rows and rows[0]["classe"] in {"buraco", "trinca"}
    first = rows[0]
    assert client.get(f"/detections/{first['id']}").json() == first
    assert client.get("/detections/nao_existe").status_code == 404
    assert client.get("/detections", params={"classe": "inexistente"}).json() == []

    ranking = client.get("/ranking").json()
    scores = [r["score_severidade"] for r in ranking]
    assert scores == sorted(scores, reverse=True)

    stats = client.get("/stats").json()
    assert stats["total_danos"] == len(rows) and stats["imagens_processadas"] == 1

    steps = client.get(f"/steps/{first['imagem_id']}").json()
    assert [s["etapa"] for s in steps][0] == "00_original" and steps[-1]["etapa"] == "09_overlay"
    for step in steps:
        response = client.get(step["url"])
        assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"

    assert len(client.get("/runs").json()) == 1
    tickets = client.get("/tickets").json()
    assert len(tickets) == sum(r["severidade"] == "alta" for r in rows)
    assert client.get("/severity-formula").json()["pesos"]["w_area"] == 0.5


def test_api_never_serves_raw_images(settings, raw_image):
    client = TestClient(create_app(settings))
    assert client.get("/files/raw/rua_teste.jpg").status_code == 404
    assert client.get("/files/intermediate/../raw/rua_teste.jpg").status_code == 404


def test_ui_renders_without_errors(settings, raw_image, monkeypatch):
    import cv2
    from streamlit.testing.v1 import AppTest

    # segunda imagem (espelhada) para testar a troca de detecção entre imagens diferentes
    cv2.imwrite(str(raw_image.with_name("rua_teste_2.jpg")), cv2.flip(cv2.imread(str(raw_image)), 1))
    run_once(settings)
    monkeypatch.setenv("DATA_DIR", str(settings.paths.data))
    monkeypatch.delenv("RUN_SCHEDULER_IN_UI", raising=False)
    app = AppTest.from_file("../src/ui/app.py", default_timeout=60).run()
    assert not app.exception
    assert app.metric[0].label == "Danos detectados" and int(app.metric[0].value) >= 1
    assert [m.label for m in app.metric] == [
        "Danos detectados", "Severidade alta", "Chamados abertos", "Imagens guardadas", "Última execução"
    ]
    assert app.metric[3].value == "2 de 200"  # imagens guardadas / limite da retenção
    assert not app.warning  # execução sem erros não mostra aviso

    # ranking: colunas legíveis e miniatura embutida em cada linha
    ranking = app.dataframe[0].value
    assert list(ranking.columns) == [
        "Foto", "Imagem", "Tipo", "Severidade", "Score", "Confiança", "Área", "Local", "Capturada em", "Fonte"
    ]
    assert ranking["Foto"].str.startswith("data:image/jpeg;base64,").all()
    assert ranking["Local"].iloc[0] == "-23.56140, -46.65590"

    # aba Transformações: imagens embutidas na grade e aviso de carregamento já removido
    def html_block(marker: str) -> str:
        return next(el.proto.body for el in app.get("html") if marker in el.proto.body)

    def gallery_html() -> str:
        return html_block('class="zcv-steps"')

    assert "data:image/jpeg;base64," in html_block('class="zcv-photo"')  # foto anotada em destaque
    assert "Severidade" in html_block('class="zcv-facts"')

    assert gallery_html().count("data:image/jpeg;base64,") >= 2
    assert not any("Carregando as transformações" in el.value for el in app.info)

    # o seletor mostra nomes legíveis (número da imagem, classe, severidade, data e hora)
    detections = app.selectbox(key="det")
    assert all(re.fullmatch(r"Imagem \d+( · dano \d+)? · \w+ \(\w+\) · \d\d/\d\d/\d{4} \d\d:\d\d", o)
               for o in detections.options), detections.options
    assert {o.split(" · ")[0] for o in detections.options} == {"Imagem 1", "Imagem 2"}

    # trocar para uma detecção de OUTRA imagem troca a grade inteira
    rows = output.read_detections(settings.paths)
    first_image = next(r["imagem_id"] for r in rows if r["id"] == detections.value)
    other = next(r["id"] for r in rows if r["imagem_id"] != first_image)
    before = gallery_html()
    detections.select(other).run()
    assert not app.exception and gallery_html() != before
    assert not any("Carregando as transformações" in el.value for el in app.info)


def test_ui_warns_when_the_last_run_had_errors(settings, monkeypatch):
    from streamlit.testing.v1 import AppTest

    (settings.paths.raw / "quebrada.jpg").write_bytes(b"nao sou uma imagem")
    run_once(settings)
    monkeypatch.setenv("DATA_DIR", str(settings.paths.data))
    monkeypatch.delenv("RUN_SCHEDULER_IN_UI", raising=False)
    app = AppTest.from_file("../src/ui/app.py", default_timeout=60).run()
    assert not app.exception
    assert len(app.warning) == 1 and "quebrada.jpg" in app.warning[0].value
    assert app.metric[3].value == "0 de 200"
    assert any("Nenhum dano registrado ainda" in el.value for el in app.info)  # estado vazio orienta
