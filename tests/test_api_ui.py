from fastapi.testclient import TestClient

from src.api.main import create_app
from src.pipeline import run_once


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

    # aba Transformações: imagens embutidas na grade e aviso de carregamento já removido
    def gallery_html() -> str:
        return next(el.proto.body for el in app.get("html") if "zcv-steps" in el.proto.body)

    assert gallery_html().count("data:image/jpeg;base64,") >= 2
    assert not any("Carregando as transformações" in el.value for el in app.info)

    # trocar para uma detecção de OUTRA imagem troca a grade inteira
    detections = app.selectbox(key="det")
    first_image = detections.value.rsplit("_", 1)[0]
    other = next(o for o in detections.options if o.rsplit("_", 1)[0] != first_image)
    before = gallery_html()
    detections.select(other).run()
    assert not app.exception and gallery_html() != before
    assert not any("Carregando as transformações" in el.value for el in app.info)
