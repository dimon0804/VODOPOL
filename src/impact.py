"""Последствия паводка в натуральных единицах, а не только в рублях.

Зачем это нужно. Рубль — не единственный язык, на котором разговаривают о
паводке, и часто не самый понятный. Дежурному МЧС важнее, сколько школ и клиник
под водой; главе района — сколько гектаров залито; финансисту — во что обойдётся
компенсация. Одно и то же событие поэтому описывается здесь тремя способами
сразу, и ни один из них не «главнее»: это разные проекции одного расчёта.

Важно, что все три считаются из того же вероятностного растра и тех же десяти
объектов, что и рублёвый ущерб. Никакой отдельной модели «в штуках» нет — иначе
числа в разных единицах начали бы расходиться между собой, и любой вопрос жюри
про несостыковку был бы законным.

Ключевое понятие — **ожидаемое число**, а не «число». Объект с вероятностью
затопления 0,4 даёт 0,4 ожидаемого объекта, а не ноль и не единицу. Это то же
самое усреднение, что и в ожидаемом ущербе EL = p × V × q, просто без денег:
считать «затоплено 3 школы», когда модель не уверена ни в одной, значит выдавать
предположение за факт.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, Sequence

import numpy as np

from src import contracts as C

#: Сколько квадратных метров в одном пикселе Sentinel-1 в нашей сетке.
#: Шаг сетки 10 м, значит пиксель — сотка. Через площадь чипа это проверяется:
#: 512 × 512 × 100 м² = 26,2 км², что совпадает с номинальной площадью чипа.
PIXEL_AREA_M2 = 100.0
PIXEL_AREA_KM2 = PIXEL_AREA_M2 / 1_000_000
PIXEL_AREA_HA = PIXEL_AREA_M2 / 10_000

#: Единицы, в которых панель умеет показывать одно и то же событие.
UNITS = (
    {
        "key": "rub",
        "title": "рубли",
        "short": "₽",
        "note": "ожидаемый прямой ущерб: p × V × q по каждому объекту",
    },
    {
        "key": "objects",
        "title": "объекты",
        "short": "об.",
        "note": "ожидаемое число пострадавших объектов: сумма вероятностей",
    },
    {
        "key": "area",
        "title": "площадь",
        "short": "км²",
        "note": "ожидаемая площадь затопления: сумма вероятностей по пикселям",
    },
)


def _num(value: Any) -> float | None:
    """Число из строки CSV; пусто остаётся пустым, а не превращается в ноль."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if np.isfinite(result) else None


def flooded_area(prob: np.ndarray, threshold: float) -> dict[str, Any]:
    """Площадь затопления двумя способами: ожидаемая и по маске.

    Ожидаемая площадь — сумма вероятностей по валидным пикселям, умноженная на
    площадь пикселя. Она честнее маски: маска на пороге 0,15 либо засчитывает
    пиксель целиком, либо выбрасывает целиком, а сумма вероятностей учитывает
    и неуверенные пиксели пропорционально уверенности.

    Площадь по маске считается рядом, потому что именно её человек видит на
    карте, и расхождение между двумя числами само по себе информативно: если
    маска сильно больше ожидаемой площади, значит порог набрал много слабых
    пикселей.
    """
    valid = prob != C.PROB_NODATA
    values = np.clip(prob[valid], 0.0, 1.0)
    expected_px = float(values.sum())
    mask_px = int((values >= threshold).sum())
    total_px = int(valid.sum())
    return {
        "expected_km2": round(expected_px * PIXEL_AREA_KM2, 4),
        "expected_ha": round(expected_px * PIXEL_AREA_HA, 2),
        "mask_km2": round(mask_px * PIXEL_AREA_KM2, 4),
        "mask_ha": round(mask_px * PIXEL_AREA_HA, 2),
        "chip_km2": round(total_px * PIXEL_AREA_KM2, 4),
        "share_of_chip": round(expected_px / total_px, 6) if total_px else None,
        "valid_pixels": total_px,
        "note": (
            "Ожидаемая площадь — сумма вероятностей по валидным пикселям; площадь "
            "по маске — те же пиксели с отсечением по порогу. Первая учитывает "
            "неуверенные пиксели пропорционально, вторая считает их целиком или никак."
        ),
    }


def by_class(asset_rows: Sequence[dict], assets: Iterable[Any]) -> list[dict[str, Any]]:
    """Разбивка последствий по типам объектов.

    Эксперты на чекпоинте просили прямо: расписать повреждение зданий, школ,
    садиков. В портфеле десять объектов десяти разных типов, поэтому разбивка —
    это и есть ответ на вопрос «что именно пострадает».
    """
    class_of = {a.asset_id: a.asset_class for a in assets}
    titles = {t.asset_class: t.title_ru for t in C.ASSET_TYPES}
    spec = {t.asset_class: t for t in C.ASSET_TYPES}

    buckets: dict[str, dict[str, Any]] = {}
    for row in asset_rows:
        asset_class = class_of.get(row["asset_id"], "")
        bucket = buckets.setdefault(
            asset_class,
            {
                "asset_class": asset_class,
                "title_ru": titles.get(asset_class, asset_class),
                "count": 0,
                "assessed": 0,
                "expected_objects": 0.0,
                "expected_loss_rub": 0.0,
                "value_rub": 0,
                "vulnerability": None,
                "asset_ids": [],
            },
        )
        bucket["count"] += 1
        bucket["asset_ids"].append(row["asset_id"])
        item = spec.get(asset_class)
        if item is not None:
            bucket["value_rub"] = item.value_rub
            bucket["vulnerability"] = item.vulnerability

        probability = _num(row.get("p_flood"))
        loss = _num(row.get("expected_loss_rub"))
        if probability is None or loss is None:
            # Объект без оценки не даёт ни нуля, ни единицы: он просто не оценён,
            # и в сумме по типу это видно отдельным счётчиком.
            continue
        bucket["assessed"] += 1
        bucket["expected_objects"] += probability
        bucket["expected_loss_rub"] += loss

    result = list(buckets.values())
    for bucket in result:
        bucket["expected_objects"] = round(bucket["expected_objects"], 4)
        bucket["expected_loss_rub"] = round(bucket["expected_loss_rub"], 2)
        bucket["not_assessed"] = bucket["count"] - bucket["assessed"]
    result.sort(key=lambda b: b["expected_loss_rub"], reverse=True)
    return result


def summarize(
    asset_rows: Sequence[dict],
    assets: Iterable[Any],
    prob: np.ndarray,
    threshold: float,
    total_loss_rub: float | Decimal,
) -> dict[str, Any]:
    """Одно событие в трёх единицах сразу плюс разбивка по типам объектов."""
    assets = list(assets)
    probabilities = [
        p for p in (_num(row.get("p_flood")) for row in asset_rows) if p is not None
    ]
    expected_objects = float(sum(probabilities))
    area = flooded_area(prob, threshold)

    return {
        "units": [dict(unit) for unit in UNITS],
        "totals": {
            # Компенсация и ущерб здесь одно и то же число, и это осознанно: мы
            # не знаем реальных правил выплат, а выдумывать коэффициент возмещения
            # значит подменить расчёт догадкой. Поле названо отдельно, потому что
            # на защите про компенсацию спрашивают именно этим словом.
            "expected_loss_rub": float(total_loss_rub),
            "compensation_rub": float(total_loss_rub),
            "expected_objects": round(expected_objects, 4),
            "assessed_objects": len(probabilities),
            "total_objects": len(asset_rows),
            "expected_area_km2": area["expected_km2"],
            "expected_area_ha": area["expected_ha"],
            "mask_area_km2": area["mask_km2"],
        },
        "area": area,
        "by_class": by_class(asset_rows, assets),
        "note": (
            "Все три единицы посчитаны из одного и того же вероятностного растра и "
            "одного портфеля объектов. «Ожидаемое число объектов» — это сумма "
            "вероятностей, а не счёт по маске: объект с вероятностью 0,4 даёт 0,4 "
            "ожидаемого объекта. Округлять его до целого нельзя — получится "
            "предположение, выданное за факт."
        ),
    }


def _chip_count(value: Any) -> int | None:
    """Число чипов в проверке. В файле метрик это бывает и списком, и счётчиком."""
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def percent_quality(metrics: dict[str, Any], scope: str = "main_flood") -> dict[str, Any]:
    """Качество карты в процентах — так, как его спрашивает жюри.

    На чекпоинте прозвучал вопрос «а какая точность в процентах». Доли единицы,
    которыми оперируют метрики сегментации, на слух не воспринимаются, поэтому
    здесь те же самые числа переводятся в проценты — без пересчёта и без
    изменения смысла, только умножением на сто.

    Сознательно не выводим accuracy: на снимке, где вода занимает проценты
    площади, «точность» в смысле доли верно классифицированных пикселей близка
    к 95 % у любого алгоритма, включая тот, что всегда отвечает «суши нет воды».
    Такое число вводит в заблуждение, и называть его точностью нечестно.
    """
    totals = metrics.get("totals", {})
    row = totals.get(scope)
    if not isinstance(row, dict):
        return {}

    def pct(name: str) -> float | None:
        value = row.get(name)
        if value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
            return None
        return round(float(value) * 100, 1)

    return {
        "scope": scope,
        "part": metrics.get("part"),
        "chips": _chip_count(metrics.get("chips")),
        "f1_pct": pct("f1"),
        "iou_pct": pct("iou"),
        "precision_pct": pct("precision"),
        "recall_pct": pct("recall"),
        "pixels": row.get("n_pixels"),
        "titles": {
            "precision_pct": (
                "из того, что названо затоплением, действительно затоплено"
            ),
            "recall_pct": "из того, что затоплено, найдено моделью",
            "f1_pct": "сводная оценка полноты и точности",
            "iou_pct": "доля совпадения найденной и настоящей области",
        },
        "note": (
            "Проценты — те же метрики сегментации, умноженные на сто. Долю верно "
            "классифицированных пикселей (accuracy) не приводим сознательно: вода "
            "занимает малую часть снимка, и по этой метрике алгоритм «воды нет» "
            "получил бы больше девяноста процентов."
        ),
    }
