"""Проверки прогона комплекта: то, что легко сломать молча.

Здесь не считается качество модели — за это отвечают метрики. Здесь проверяется
подпись результата: дата съёмки и сроки, которые постановка требует показывать
рядом с идентификатором чипа.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.cli.run_bundle import EVENT_ALIASES, event_date, scenario_dates
from src.data import fetch

#: Все одиннадцать размеченных событий набора.
EVENTS = (
    "Bolivia",
    "Ghana",
    "India",
    "Mekong",
    "Nigeria",
    "Pakistan",
    "Paraguay",
    "Somalia",
    "Spain",
    "Sri-Lanka",
    "USA",
)


@pytest.fixture(scope="module", autouse=True)
def _metadata_present() -> None:
    path = fetch.DEFAULT_ROOT / "Sen1Floods11_Metadata.geojson"
    if not path.exists():
        pytest.skip("метаданные набора не выкачаны: python -m src.cli.fetch_data")


def test_дата_съёмки_известна_для_всех_событий() -> None:
    """Постановка требует показывать дату чипа, значит прочерков быть не должно.

    Ровно на этом ловился Mekong: чипы называют событие Mekong, а метаданные
    набора — Cambodia, и дата тихо не находилась у тридцати чипов.
    """
    missing = [event for event in EVENTS if event_date(event) is None]
    assert not missing, f"дата съёмки не найдена: {missing}"


def test_алиас_события_ведёт_к_реальной_дате() -> None:
    assert "mekong" in EVENT_ALIASES
    assert event_date("Mekong") == date(2018, 8, 5)


def test_даты_событий_лежат_в_эпохе_sentinel1() -> None:
    """Sentinel-1A запущен в 2014-м: дата раньше означала бы разбор мусора."""
    for event in EVENTS:
        observed = event_date(event)
        assert observed is not None, event
        assert date(2014, 4, 3) <= observed <= date.today(), (event, observed)


def test_срок_решения_не_раньше_съёмки() -> None:
    """Решение о доснимке принимается после того, как пришёл исходный снимок."""
    for event in EVENTS:
        dates = scenario_dates(event, "")
        observed = dates.get("event_date", "")
        deadline = dates.get("decision_deadline", "")
        assert observed and deadline, event
        assert deadline >= observed, (event, observed, deadline)
