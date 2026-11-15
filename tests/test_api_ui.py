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

    assert "data:image/jpeg;base64," in html_block('class="zcv-photo"')  # foto com o dano em destaque
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
    assert any("Nenhuma imagem analisada ainda" in el.value for el in app.info)  # estado vazio orienta


def test_ui_lists_images_without_damage_and_highlights_the_chosen_damage(settings, raw_image, monkeypatch):
    import json

    import cv2
    from streamlit.testing.v1 import AppTest

    from src.model.infer import Detection

    class _Two:
        name = "fake"

        def detect(self, img, baseline_mask=None):
            return [
                Detection("buraco", 0.9, (150, 300, 200, 120), 200 * 120),
                Detection("trinca", 0.6, (400, 200, 120, 60), 120 * 60),
            ]

    class _Nothing:
        name = "fake"

        def detect(self, img, baseline_mask=None):
            return []

    run_once(settings, detector=_Two())  # Imagem 1: dois danos, com localização
    clean = raw_image.with_name("limpa.jpg")  # Imagem 2: sem dano, com localização
    cv2.imwrite(str(clean), cv2.flip(cv2.imread(str(raw_image)), 1))
    clean.with_suffix(".json").write_text(json.dumps({"lat": -23.57, "lon": -46.66}), encoding="utf-8")
    run_once(settings, detector=_Nothing())

    monkeypatch.setenv("DATA_DIR", str(settings.paths.data))
    monkeypatch.delenv("RUN_SCHEDULER_IN_UI", raising=False)
    app = AppTest.from_file("../src/ui/app.py", default_timeout=60).run()
    assert not app.exception

    def html_block(marker: str) -> str:
        return next(el.proto.body for el in app.get("html") if marker in el.proto.body)

    # o seletor lista os danos e, depois, a imagem sem dano
    selector = app.selectbox(key="det")
    assert [o.split(" · ")[:2] for o in selector.options] == [
        ["Imagem 1", "dano 1"], ["Imagem 1", "dano 2"], ["Imagem 2", "sem dano"]
    ]
    assert any("1 imagem analisada não teve dano detectado" in c.value for c in app.caption)

    # trocar entre dois danos da MESMA imagem muda a foto (o escolhido fica em destaque)
    rows = output.read_detections(settings.paths)
    other = next(r["id"] for r in rows if r["id"] != selector.value)
    first_photo = html_block('class="zcv-photo"')
    # caixas desenhadas em HTML sobre a foto: o escolhido com rótulo completo (com acento),
    # o outro dano tracejado
    assert first_photo.count('class="zcv-box is-chosen"') == 1 and first_photo.count('class="zcv-box is-other"') == 1
    assert "Buraco · 90% · Alta" in first_photo
    selector.select(other).run()
    second_photo = html_block('class="zcv-photo"')
    assert not app.exception and second_photo != first_photo
    assert "Trinca · 60% · " in second_photo
    # a foto vai sem recompressão: os mesmos bytes do arquivo original
    import base64

    image_id = next(r["imagem_id"] for r in rows)
    original = (settings.paths.intermediate / "00_original" / f"{image_id}.jpg").read_bytes()
    assert base64.b64encode(original).decode() in second_photo

    # a imagem sem dano também pode ser inspecionada
    app.selectbox(key="det").select("img:" + next(im["imagem_id"] for im in output.read_images(settings.paths) if not im["danos"])).run()
    assert not app.exception
    assert "Nenhum dano detectado" in html_block('class="zcv-facts"')
    assert 'class="zcv-steps"' in html_block('class="zcv-steps"')

    # mapa: os dois danos e a imagem sem dano (ponto cinza)
    deck = json.loads(app.get("deck_gl_json_chart")[0].proto.json)
    points = deck["layers"][0]["data"]
    # raio em pixels (texto literal): como expressão, o pydeck desenha um ponto do tamanho do mapa
    assert deck["layers"][0]["radiusUnits"] == "pixels"
    assert [pt["detalhe"] for pt in points].count("Sem dano detectado") == 1 and len(points) == 3

    # a escolha continua a mesma quando chega uma detecção nova
    chosen = app.selectbox(key="det").value
    new = raw_image.with_name("nova.jpg")
    cv2.imwrite(str(new), cv2.flip(cv2.imread(str(raw_image)), 0))
    run_once(settings, detector=_Two())
    app.run()
    assert not app.exception and app.selectbox(key="det").value == chosen
    assert len(app.selectbox(key="det").options) == 5
