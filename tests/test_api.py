"""Проверки слоя сервиса: HTTP-контракт панели оператора.

Тесты не проверяют сами числа — за это отвечают тесты расчётной части. Здесь
проверяется, что панель получает то, на что сверстана: коды ответов, состав
полей, границы карты в заголовке и понятные ошибки вместо трейсбеков.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import config
from src.api.app import RunState, create_app
from src.runtime import RunContext

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _real_run_dir() -> Path | None:
    """Самый свежий готовый комплект, если он собран."""
    runs = sorted(
        (PROJECT_ROOT / "outputs").glob("*/" + config.RUN_MARKER),
        key=lambda p: p.stat().st_mtime,
    )
    return runs[-1].parent if runs else None


@pytest.fixture(scope="module")
def run_dir() -> Path:
    found = _real_run_dir()
    if found is None:
        pytest.skip("готовый комплект не собран: python -m src.cli.run_bundle")
    return found


@pytest.fixture(scope="module")
def client(run_dir: Path) -> TestClient:
    state = RunState(
        ctx=RunContext.load(run_dir),
        run_dir=run_dir,
        source="env",
        message="тестовый комплект",
    )
    return TestClient(create_app(state))


@pytest.fixture(scope="module")
def demo_client() -> TestClient:
    state = RunState(ctx=RunContext.demo(), run_dir=None, source="demo", message="демо")
    return TestClient(create_app(state))


# ── состояние ────────────────────────────────────────────────────────────────


def test_health_отвечает(client: TestClient, run_dir: Path) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == config.API_VERSION
    assert body["demo"] is False
    assert body["run_id"] == run_dir.name
    assert body["chip_id"]


def test_health_в_демо_режиме_помечен(demo_client: TestClient) -> None:
    body = demo_client.get("/api/health").json()
    assert body["status"] == "demo"
    assert body["demo"] is True


def test_отсутствующий_комплект_это_404_с_объяснением() -> None:
    state = RunState(ctx=None, run_dir=Path("outputs/нет-такого"), source="env",
                     message="комплект не найден, соберите его")
    client = TestClient(create_app(state))
    assert client.get("/api/health").status_code == 200  # здоровье отвечает всегда
    response = client.get("/api/run")
    assert response.status_code == 404
    assert "комплект" in response.json()["error"].lower()


# ── данные ───────────────────────────────────────────────────────────────────


def test_run_отдаёт_chip_id_и_границы(client: TestClient) -> None:
    body = client.get("/api/run").json()
    assert body["chip_id"]
    assert body["run_id"]
    assert body["demo"] is False
    assert body["base_rate_status"] in {"scenario", "official"}
    bounds = body["bounds"]
    assert isinstance(bounds, list) and len(bounds) == 4
    assert bounds == body["raster_bounds"]
    assert bounds[0] < bounds[2] and bounds[1] < bounds[3]


def test_assets_ровно_десять_фич(client: TestClient) -> None:
    body = client.get("/api/assets").json()
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) == 10
    for feature in body["features"]:
        props = feature["properties"]
        assert props["asset_id"]
        assert props["status"] in {"ok", "partial", "no_data"}
        if props["status"] != "ok":
            # Пустое остаётся пустым: ноль здесь был бы ложью.
            assert props["p_flood"] is None
            assert props["expected_loss_rub"] is None
            assert props["rank"] is None


def test_candidates_это_геоджейсон_зон(client: TestClient) -> None:
    body = client.get("/api/candidates").json()
    assert body["type"] == "FeatureCollection"
    assert body["features"]
    props = body["features"][0]["properties"]
    for field in ("candidate_id", "area_km2", "base_rate_rub_km2", "base_rate_status"):
        assert field in props


def test_sensitivity_таблица_непустая(client: TestClient) -> None:
    body = client.get("/api/sensitivity").json()
    assert isinstance(body, list) and body
    assert {"scenario_id", "strategy", "result_status"} <= set(body[0])


# ── стратегии ────────────────────────────────────────────────────────────────


def test_strategies_без_бюджета_берёт_объявленный(client: TestClient) -> None:
    body = client.get("/api/strategies").json()
    assert set(body["plans"]) == {"A", "B", "C"}
    codes = {row["strategy"] for row in body["comparison"]}
    assert codes == {"A", "B", "C"}


def test_разный_бюджет_даёт_разный_состав_C(client: TestClient) -> None:
    small = client.get("/api/strategies", params={"budget": 5000})
    large = client.get("/api/strategies", params={"budget": 45000})
    assert small.status_code == 200 and large.status_code == 200

    small_plan = small.json()["plans"]["C"]
    large_plan = large.json()["plans"]["C"]
    assert small_plan != large_plan
    assert len(small_plan) < len(large_plan)
    # Портфель и правило B от бюджета не зависят: меняется только корзина заказа.
    assert small.json()["plans"]["B"] == large.json()["plans"]["B"]


def test_бюджет_попадает_в_сравнение(client: TestClient) -> None:
    body = client.get("/api/strategies", params={"budget": 5000}).json()
    for row in body["comparison"]:
        assert float(row["budget_rub"]) == pytest.approx(5000.0)
        assert row["uncertainty_status"] in {
            "baseline", "scenario", "measured", "not_estimated",
        }


def test_отрицательный_бюджет_отвергается(client: TestClient) -> None:
    assert client.get("/api/strategies", params={"budget": -1}).status_code == 422


# ── карта ────────────────────────────────────────────────────────────────────


def test_probability_png_с_границами(client: TestClient) -> None:
    response = client.get("/api/raster/probability.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(PNG_MAGIC)

    bounds = json.loads(response.headers["X-Bounds"])
    assert len(bounds) == 4
    assert bounds[0] < bounds[2] and bounds[1] < bounds[3]
    assert bounds == client.get("/api/run").json()["bounds"]


def test_маска_и_неопределённость_тоже_png(client: TestClient) -> None:
    for kind in ("mask", "uncertainty"):
        response = client.get(f"/api/raster/{kind}.png")
        assert response.status_code == 200, kind
        assert response.content.startswith(PNG_MAGIC), kind
        assert "X-Bounds" in response.headers, kind


def test_несуществующий_слой_это_понятная_ошибка(client: TestClient) -> None:
    response = client.get("/api/raster/rainbow.png")
    assert response.status_code in (400, 404)
    error = response.json()["error"]
    assert "rainbow" in error
    assert "probability" in error  # перечислены допустимые значения
    assert "Traceback" not in error


def test_недоступный_слой_объясняется_словами(client: TestClient) -> None:
    """Подложка S1 может быть недоступна — это должно быть понятным ответом."""
    response = client.get("/api/raster/s1.png")
    assert response.status_code in (200, 404, 503)
    if response.status_code != 200:
        error = response.json()["error"]
        assert "S1" in error or "s1" in error
        assert "Traceback" not in error


def test_демо_отдаёт_все_слои(demo_client: TestClient) -> None:
    for kind in config.RASTER_KINDS:
        response = demo_client.get(f"/api/raster/{kind}.png")
        assert response.status_code == 200, kind
        assert response.content.startswith(PNG_MAGIC), kind


# ── выгрузка и статика ───────────────────────────────────────────────────────


def test_бандл_это_zip_архив(client: TestClient) -> None:
    response = client.get("/api/bundle.zip")
    assert response.status_code == 200
    assert response.content[:2] == b"PK"
    assert "attachment" in response.headers["content-disposition"]


def test_страница_панели_отдаётся(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Водополь" in response.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200
