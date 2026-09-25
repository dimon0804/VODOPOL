"""Портфель из десяти объектов и ожидаемый прямой ущерб.

Требования кейса, каждое из которых проверяется жюри отдельно:

* ровно десять точечных объектов, по одному каждого типа из таблицы;
* координаты случайны, с фиксированным зерном, в валидной области ИСХОДНОГО S1 —
  без обращения к прогнозу, к карте ущерба и к ручным меткам. Метки здесь не
  используются сознательно: подбор мест ради красивого ущерба запрещён;
* попал в nodata — повтор по заранее зафиксированному правилу;
* EL = p × V × q, где p берётся из вероятностного растра независимо от бинарной
  маски. Низкое p не обнуляет ущерб и не исключает объект из расчёта;
* у статусов partial и no_data поля p, EL и rank остаются пустыми, а не нулевыми.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src import contracts as C

#: Правило размещения фиксируется строкой и уходит в run_metadata.json как есть.
PLACEMENT_RULE = (
    "Равномерная выборка пикселя по всей сетке чипа генератором numpy PCG64 с "
    "фиксированным зерном. Пиксель принимается, если оба канала S1 в нём конечны "
    "(валидная область исходного снимка) и он ещё не занят другим объектом; иначе "
    "выборка повторяется. Ручные метки, вероятностный растр и карта ущерба при "
    "размещении не используются. Объекты назначаются типам в порядке таблицы кейса."
)

#: Правило извлечения вероятности — тоже часть паспорта запуска.
PROBABILITY_RULE = (
    "p_flood берётся из flood_probability.tif в пикселе, содержащем координату объекта, "
    "без сглаживания и независимо от бинарной маски."
)

MAX_ATTEMPTS_PER_ASSET = 10_000
#: Доля nodata в окрестности 3×3, начиная с которой объект считается ненадёжно оценённым.
PARTIAL_NODATA_SHARE = 1 / 3


@dataclass(frozen=True)
class Asset:
    asset_id: str
    chip_id: str
    asset_class: str
    asset_value_rub: int
    vulnerability_coef: float
    row: int
    col: int
    longitude: float
    latitude: float

    def as_row(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "chip_id": self.chip_id,
            "asset_class": self.asset_class,
            "asset_value_rub": self.asset_value_rub,
            "vulnerability_coef": self.vulnerability_coef,
            "longitude": self.longitude,
            "latitude": self.latitude,
        }

    def as_feature(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [self.longitude, self.latitude]},
            "properties": {
                "asset_id": self.asset_id,
                "chip_id": self.chip_id,
                "asset_class": self.asset_class,
                "asset_value_rub": self.asset_value_rub,
                "vulnerability_coef": self.vulnerability_coef,
            },
        }


def s1_valid_area(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """Валидная область исходного снимка: оба канала конечны.

    Ручная метка сюда не входит намеренно — постановка требует размещать объекты в
    валидной области S1, а не там, где кто-то уже разметил воду.
    """
    return np.isfinite(vv) & np.isfinite(vh)


def pixel_to_lonlat(transform, row: int, col: int) -> tuple[float, float]:
    """Центр пикселя в градусах. Порядок возвращаемых значений — долгота, широта."""
    lon, lat = transform * (col + 0.5, row + 0.5)
    return float(lon), float(lat)


def place_assets(
    chip_id: str,
    vv: np.ndarray,
    vh: np.ndarray,
    transform,
    seed: int,
) -> list[Asset]:
    """Размещает ровно десять объектов по зафиксированному правилу."""
    valid = s1_valid_area(vv, vh)
    if not valid.any():
        raise ValueError(f"в чипе {chip_id} нет валидной области S1")

    height, width = valid.shape
    rng = np.random.default_rng(seed)
    taken: set[tuple[int, int]] = set()
    assets: list[Asset] = []

    for index, asset_type in enumerate(C.ASSET_TYPES, start=1):
        for attempt in range(MAX_ATTEMPTS_PER_ASSET):
            row = int(rng.integers(0, height))
            col = int(rng.integers(0, width))
            if not valid[row, col] or (row, col) in taken:
                continue  # nodata или занятая точка — повторяем выборку
            taken.add((row, col))
            lon, lat = pixel_to_lonlat(transform, row, col)
            assets.append(
                Asset(
                    asset_id=f"a{index:02d}",
                    chip_id=chip_id,
                    asset_class=asset_type.asset_class,
                    asset_value_rub=asset_type.value_rub,
                    vulnerability_coef=asset_type.vulnerability,
                    row=row,
                    col=col,
                    longitude=lon,
                    latitude=lat,
                )
            )
            break
        else:
            raise RuntimeError(
                f"не удалось разместить объект {asset_type.asset_class} за "
                f"{MAX_ATTEMPTS_PER_ASSET} попыток — валидная область слишком мала"
            )

    if len(assets) != C.ASSET_COUNT:
        raise AssertionError(f"размещено {len(assets)} объектов вместо {C.ASSET_COUNT}")
    return assets


def _status_at(prob: np.ndarray, row: int, col: int) -> str:
    """Статус оценки объекта по окрестности вероятностного растра."""
    if prob[row, col] == C.PROB_NODATA or not np.isfinite(prob[row, col]):
        return C.ASSET_STATUS_NO_DATA
    height, width = prob.shape
    r0, r1 = max(row - 1, 0), min(row + 2, height)
    c0, c1 = max(col - 1, 0), min(col + 2, width)
    window = prob[r0:r1, c0:c1]
    bad = (window == C.PROB_NODATA) | ~np.isfinite(window)
    if window.size and bad.sum() / window.size > PARTIAL_NODATA_SHARE:
        return C.ASSET_STATUS_PARTIAL
    return C.ASSET_STATUS_OK


def evaluate_assets(
    assets: Sequence[Asset],
    prob: np.ndarray,
    uncertainty: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Строки asset_loss.csv: вероятность, ущерб, неопределённость, ранг, статус.

    Ранг присваивается только оценённым объектам: у partial и no_data он пуст, потому
    что ранжировать нечего. Неоценённая экспозиция при этом не исчезает — она
    показывается отдельно в итогах и в интерфейсе.
    """
    rows: list[dict[str, Any]] = []
    for asset in assets:
        status = _status_at(prob, asset.row, asset.col)
        if status == C.ASSET_STATUS_OK:
            p = float(prob[asset.row, asset.col])
            el = C.expected_loss(p, asset.asset_value_rub, asset.vulnerability_coef)
            unc = (
                float(uncertainty[asset.row, asset.col])
                if uncertainty is not None and np.isfinite(uncertainty[asset.row, asset.col])
                else None
            )
        else:
            p, el, unc = None, None, None
        rows.append(
            {
                "asset_id": asset.asset_id,
                "chip_id": asset.chip_id,
                "p_flood": p,
                "expected_loss_rub": el,
                "uncertainty": unc,
                "rank": None,
                "status": status,
            }
        )

    ranked = sorted(
        (row for row in rows if row["status"] == C.ASSET_STATUS_OK),
        key=lambda row: (-row["expected_loss_rub"], row["asset_id"]),
    )
    for position, row in enumerate(ranked, start=1):
        row["rank"] = position
    return rows


def total_expected_loss(rows: Sequence[dict[str, Any]]) -> float:
    """Суммарный оценённый ущерб. Неоценённые объекты в сумму не входят."""
    return float(sum(row["expected_loss_rub"] or 0.0 for row in rows))


def unassessed_exposure(rows: Sequence[dict[str, Any]], assets: Sequence[Asset]) -> dict[str, Any]:
    """Экспозиция, которая осталась без оценки. Показывается рядом с итогом."""
    by_id = {asset.asset_id: asset for asset in assets}
    missing = [row["asset_id"] for row in rows if row["status"] != C.ASSET_STATUS_OK]
    return {
        "asset_ids": missing,
        "count": len(missing),
        "value_rub": float(sum(by_id[a].asset_value_rub for a in missing)),
        "note": "Объекты без надёжной оценки: ущерб не равен нулю, он неизвестен.",
    }


def proxy_error(
    rows: Sequence[dict[str, Any]],
    assets: Sequence[Asset],
    truth: np.ndarray,
    label_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Прокси-проверка ущерба: EL = p×V×q против EL_ref = y×V×q при тех же V и q.

    Контрольные точки не подбираются по прогнозу — берутся те же самые объекты,
    размещённые случайно до всякого расчёта. Это оценка качества перехода
    «вероятность → рубли», а не измерение фактических потерь.

    `label_valid` — маска пикселей с надёжной ручной меткой. Кейс требует считать
    прокси-ошибку на выборке с валидными метками, а пиксели со значением −1 меткой не
    являются: объект, попавший в такой пиксель, исключается из проверки и считается
    отдельно, а не записывается молча в сушу.
    """
    by_id = {asset.asset_id: asset for asset in assets}
    pairs: list[tuple[float, float]] = []
    skipped: list[str] = []
    for row in rows:
        if row["status"] != C.ASSET_STATUS_OK:
            continue
        asset = by_id[row["asset_id"]]
        if label_valid is not None and not label_valid[asset.row, asset.col]:
            skipped.append(asset.asset_id)
            continue
        raw = truth[asset.row, asset.col]
        # Если пришёл сырой слой меток, значение −1 означает «метки нет». Считать его
        # ни водой, ни сушей нельзя: bool(-1) дало бы воду, а маска затопления — сушу.
        if np.issubdtype(np.asarray(truth).dtype, np.integer) and int(raw) == C.LABEL_INVALID:
            skipped.append(asset.asset_id)
            continue
        y = float(bool(raw))
        reference = C.expected_loss(y, asset.asset_value_rub, asset.vulnerability_coef)
        pairs.append((float(row["expected_loss_rub"]), reference))

    if not pairs:
        return {
            "n_points": 0,
            "n_skipped_without_label": len(skipped),
            "skipped_asset_ids": skipped,
            "status": "not_checked",
            "note": "Пригодных контрольных точек нет, прокси-ошибка не проверена.",
        }

    predicted = np.array([p for p, _ in pairs])
    reference = np.array([r for _, r in pairs])
    return {
        "n_points": len(pairs),
        "n_skipped_without_label": len(skipped),
        "skipped_asset_ids": skipped,
        "status": "checked",
        "mae_rub": float(np.mean(np.abs(predicted - reference))),
        "rmse_rub": float(np.sqrt(np.mean((predicted - reference) ** 2))),
        "bias_rub": float(np.mean(predicted - reference)),
        "sum_predicted_rub": float(predicted.sum()),
        "sum_reference_rub": float(reference.sum()),
        "rule": "Контрольные точки — те же десять объектов, размещённые случайно до расчёта.",
    }
