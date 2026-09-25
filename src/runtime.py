"""Единственная точка входа сервиса в расчёты.

Слой API не вычисляет ничего сам: он вызывает методы `RunContext` и сериализует
то, что получил. Так цифры на экране не расходятся с файлами сданного комплекта.

Сигнатуры методов заморожены вместе с `contracts.py` — верстать можно по ним,
не дожидаясь готовых данных: `RunContext.demo()` отдаёт ту же структуру на
синтетическом растре и помечает себя `"demo": true`.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src import contracts as C

RasterKind = Literal["s1", "probability", "mask", "uncertainty"]

# Палитры подобраны так, чтобы слои читались поверх серой радарной подложки.
_PROB_STOPS = np.array(
    [
        [247, 251, 255],
        [198, 219, 239],
        [107, 174, 214],
        [33, 113, 181],
        [8, 48, 107],
    ],
    dtype=np.float32,
)
_UNC_STOPS = np.array(
    [
        [255, 247, 236],
        [253, 212, 158],
        [253, 141, 60],
        [217, 72, 1],
        [127, 39, 4],
    ],
    dtype=np.float32,
)


def _ramp(values: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Линейная интерполяция значений 0..1 по опорным цветам палитры."""
    clipped = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
    position = clipped * (len(stops) - 1)
    low = np.floor(position).astype(np.int32)
    high = np.minimum(low + 1, len(stops) - 1)
    weight = (position - low)[..., None]
    return (stops[low] * (1 - weight) + stops[high] * weight).astype(np.uint8)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding=C.CSV_ENCODING, newline="") as handle:
        return list(csv.DictReader(handle, delimiter=C.CSV_DELIMITER))


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding=C.CSV_ENCODING))


def _typed_comparison(row: dict[str, str]) -> dict[str, Any]:
    """Приводит строку сравнения из CSV к тем же типам, что даёт пересчёт.

    Одна и та же ручка сервиса не должна отдавать разные контракты. Особенно опасно
    было поле budget_feasible: строка "false" в любом условии истинна, и стратегия B,
    вышедшая за бюджет, читалась бы интерфейсом как уложившаяся.
    """
    typed: dict[str, Any] = dict(row)
    for field in (
        "data_cost_rub",
        "other_cost_rub",
        "decision_cost_rub",
        "budget_rub",
        "covered_expected_loss_rub",
        "coverage_share",
        "residual_uncertainty",
    ):
        typed[field] = _num(row.get(field))
    raw = str(row.get("budget_feasible", "")).strip().lower()
    typed["budget_feasible"] = raw in ("true", "1", "yes", "да")
    return typed


def _num(raw: str | None) -> float | None:
    """Пустое значение остаётся пустым: у partial и no_data ноль был бы ложью."""
    if raw is None or raw == C.CSV_EMPTY:
        return None
    return float(raw)


@dataclass
class RunContext:
    """Готовый комплект одного запуска: один чип, один портфель, один бюджет."""

    run_dir: Path
    is_demo: bool = False
    _demo_payload: dict[str, Any] | None = None

    # ── загрузка ─────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, run_dir: str | Path) -> "RunContext":
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(
                f"комплект запуска не найден: {run_dir}. "
                "Соберите его командой python -m src.cli.run_bundle"
            )
        return cls(run_dir=run_dir)

    @classmethod
    def demo(cls) -> "RunContext":
        """Синтетический комплект той же структуры — чтобы верстать до готовых данных."""
        return cls(run_dir=Path("outputs/_demo"), is_demo=True, _demo_payload=_build_demo())

    # ── данные для интерфейса ────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Паспорт запуска: чип, порог, бюджет, статусы, версии, границы карты."""
        if self.is_demo:
            return self._demo_payload["summary"]
        meta = _read_json(self.run_dir / C.F_RUN_METADATA)
        sources = _read_json(self.run_dir / C.F_SOURCE_MANIFEST)
        return {
            "demo": False,
            "run_id": meta.get("run_id"),
            "chip_id": meta.get("chip_id"),
            "event_id": meta.get("event_id"),
            # Путь к исходному снимку нужен сервису для подложки карты.
            "s1_path": meta.get("s1_path") or meta.get("source_chip_path"),
            "threshold": meta.get("threshold"),
            "budget_rub": meta.get("budget_rub"),
            "decision_deadline": meta.get("decision_deadline"),
            "target_class": meta.get("target_class"),
            "base_rate_status": meta.get("base_rate_status", C.BASE_RATE_STATUS_SCENARIO),
            "legal_edition": meta.get("legal_edition", C.LEGAL_EDITION),
            "methods": meta.get("methods", {}),
            "bounds": meta.get("bounds"),
            "sources": sources,
        }

    def assets(self) -> dict[str, Any]:
        """Десять объектов с ущербом. Пустые p/EL/rank у partial и no_data — так и надо."""
        if self.is_demo:
            return self._demo_payload["assets"]
        geo = _read_json(self.run_dir / C.F_ASSETS_GEOJSON)
        loss = {row["asset_id"]: row for row in _read_csv(self.run_dir / C.F_ASSET_LOSS)}
        for feature in geo.get("features", []):
            row = loss.get(feature["properties"].get("asset_id"), {})
            feature["properties"].update(
                {
                    "p_flood": _num(row.get("p_flood")),
                    "expected_loss_rub": _num(row.get("expected_loss_rub")),
                    "uncertainty": _num(row.get("uncertainty")),
                    "rank": int(row["rank"]) if row.get("rank") else None,
                    "status": row.get("status", C.ASSET_STATUS_NO_DATA),
                }
            )
        return geo

    def candidates(self) -> dict[str, Any]:
        """Каталог всех предложенных зон — не только выбранных."""
        if self.is_demo:
            return self._demo_payload["candidates"]
        return _read_json(self.run_dir / C.F_CANDIDATES)

    def strategies(self, budget_rub: float | None = None) -> dict[str, Any]:
        """A/B/C под заданный бюджет.

        Пересчитываются только отбор зон, стоимость и сравнение. Вероятности,
        портфель и порог бинаризации при этом не меняются и модель не переобучается.
        """
        if self.is_demo:
            return _demo_strategies(self._demo_payload, budget_rub)
        plans = _read_json(self.run_dir / C.F_STRATEGY_PLANS)
        positions = _read_csv(self.run_dir / C.F_PROCUREMENT)
        comparison = [_typed_comparison(row) for row in _read_csv(self.run_dir / C.F_STRATEGY_COMPARISON)]
        if budget_rub is not None:
            from src.procurement.strategies import recompute_under_budget

            plans, positions, comparison = recompute_under_budget(self, float(budget_rub))
        return {
            "budget_rub": budget_rub if budget_rub is not None else self.summary().get("budget_rub"),
            "plans": plans,
            "positions": positions,
            "comparison": comparison,
        }

    def sensitivity(self) -> list[dict[str, str]]:
        if self.is_demo:
            return self._demo_payload["sensitivity"]
        return _read_csv(self.run_dir / C.F_SENSITIVITY)

    # ── растры для карты ─────────────────────────────────────────────────────

    def raster_png(self, kind: RasterKind) -> tuple[bytes, list[float]]:
        """PNG слоя и его границы [запад, юг, восток, север] для наложения на карту."""
        from PIL import Image

        if self.is_demo:
            rgba, bounds = _demo_raster(self._demo_payload, kind)
        else:
            rgba, bounds = self._render_raster(kind)
        buffer = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), bounds

    def _render_raster(self, kind: RasterKind) -> tuple[np.ndarray, list[float]]:
        import rasterio

        if kind == "s1":
            source = Path(self.summary().get("s1_path", ""))
            if not source.exists():
                raise FileNotFoundError("исходный S1-чип недоступен, подложку не построить")
            with rasterio.open(source) as dataset:
                band = dataset.read(1).astype(np.float32)  # VV в дБ
                bounds = list(dataset.bounds)
            finite = np.isfinite(band)
            low, high = np.percentile(band[finite], [2, 98]) if finite.any() else (-25.0, 0.0)
            grey = np.clip((band - low) / max(high - low, 1e-6), 0, 1)
            rgba = np.zeros((*band.shape, 4), dtype=np.uint8)
            rgba[..., :3] = (grey * 255).astype(np.uint8)[..., None]
            rgba[..., 3] = np.where(finite, 255, 0)
            return rgba, bounds

        name = {
            "probability": C.F_PROBABILITY,
            "uncertainty": "uncertainty.tif",
            "mask": C.F_MASK,
        }[kind]
        path = self.run_dir / name
        if not path.exists():
            raise FileNotFoundError(f"слой {kind} отсутствует в комплекте: {path}")
        with rasterio.open(path) as dataset:
            band = dataset.read(1)
            bounds = list(dataset.bounds)

        rgba = np.zeros((*band.shape, 4), dtype=np.uint8)
        if kind == "mask":
            water = band == C.MASK_WATER
            nodata = band == C.MASK_NODATA
            rgba[water] = (8, 48, 107, 210)
            rgba[~water & ~nodata] = (0, 0, 0, 0)
            return rgba, bounds

        values = band.astype(np.float32)
        valid = values != C.PROB_NODATA
        stops = _PROB_STOPS if kind == "probability" else _UNC_STOPS
        rgba[..., :3] = _ramp(np.where(valid, values, 0.0), stops)
        # Прозрачность растёт вместе со значением: сухие участки не закрывают подложку.
        alpha = np.clip(np.where(valid, values, 0.0), 0, 1) * 235
        rgba[..., 3] = np.where(valid, alpha.astype(np.uint8), 0)
        return rgba, bounds

    # ── выгрузка ─────────────────────────────────────────────────────────────

    def bundle_zip(self) -> bytes:
        """Весь комплект одним архивом — кнопка выгрузки в интерфейсе."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in C.BUNDLE_FILES:
                path = self.run_dir / name
                if path.exists():
                    archive.write(path, arcname=name)
        return buffer.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Демонстрационный комплект: та же структура на синтетическом растре
# ─────────────────────────────────────────────────────────────────────────────


def _build_demo() -> dict[str, Any]:
    """Синтетика для вёрстки. Помечена demo=True и никогда не выдаётся за расчёт."""
    rng = np.random.default_rng(42)
    size = 256
    west, south = 93.735247, 26.722278
    east, north = west + size * 1.8e-4, south + size * 1.8e-4
    bounds = [west, south, east, north]

    yy, xx = np.mgrid[0:size, 0:size] / size
    river = np.exp(-(((yy - 0.55 - 0.12 * np.sin(6 * xx)) / 0.06) ** 2))
    flood = np.exp(-(((yy - 0.62) ** 2 + (xx - 0.35) ** 2) / 0.05))
    prob = np.clip(0.85 * river + 0.7 * flood + 0.05 * rng.random((size, size)), 0, 1)
    uncertainty = np.clip(0.9 * prob * (1 - prob) * 4 + 0.05, 0, 1)

    features = []
    rows = []
    for index, asset_type in enumerate(C.ASSET_TYPES):
        col, row = rng.integers(20, size - 20, size=2)
        lon = west + (col + 0.5) * (east - west) / size
        lat = north - (row + 0.5) * (north - south) / size
        p = float(prob[row, col])
        status = C.ASSET_STATUS_OK if index != 9 else C.ASSET_STATUS_NO_DATA
        el = C.expected_loss(p, asset_type.value_rub, asset_type.vulnerability)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "asset_id": f"a{index + 1:02d}",
                    "chip_id": "DEMO_000000",
                    "asset_class": asset_type.asset_class,
                    "asset_value_rub": asset_type.value_rub,
                    "vulnerability_coef": asset_type.vulnerability,
                    "p_flood": p if status == C.ASSET_STATUS_OK else None,
                    "expected_loss_rub": el if status == C.ASSET_STATUS_OK else None,
                    "uncertainty": float(uncertainty[row, col]) if status == C.ASSET_STATUS_OK else None,
                    "rank": None,
                    "status": status,
                },
            }
        )
        rows.append((index, el, status))

    ranked = sorted((r for r in rows if r[2] == C.ASSET_STATUS_OK), key=lambda r: -r[1])
    for position, (index, _, _) in enumerate(ranked, start=1):
        features[index]["properties"]["rank"] = position

    candidates = []
    for k in range(6):
        c0 = west + 0.2 * (east - west) + 0.11 * k * (east - west)
        r0 = south + 0.3 * (north - south)
        step_lon, step_lat = 0.011, 0.009
        candidates.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [c0, r0],
                            [c0 + step_lon, r0],
                            [c0 + step_lon, r0 + step_lat],
                            [c0, r0 + step_lat],
                            [c0, r0],
                        ]
                    ],
                },
                "properties": {
                    "candidate_id": f"cand_demo_{k + 1:02d}",
                    "sensor_type": C.SENSOR_SAR,
                    "resolution_m": 3,
                    "acquisition_type": "new",
                    "data_role": C.DATA_ROLE_EVENT,
                    "observation_at": "",
                    "available_at": "",
                    "processing_level": "L2",
                    "usage_type": "internal",
                    "guaranteed_purchase": True,
                    "area_km2": 1.21,
                    "base_rate_rub_km2": float(C.BASE_RATE_RUB_KM2_SCENARIO),
                    "base_rate_status": C.BASE_RATE_STATUS_SCENARIO,
                    "base_rate_source": C.BASE_RATE_SOURCE_SCENARIO,
                    "cost_rub": 2060.19 + 37.0 * k,
                    "covered_assets": [f"a{(k % 9) + 1:02d}"],
                },
            }
        )

    return {
        "summary": {
            "demo": True,
            "run_id": "demo",
            "chip_id": "DEMO_000000",
            "event_id": "DEMO",
            "threshold": 0.5,
            "budget_rub": 8000.0,
            "decision_deadline": "",
            "target_class": "временное затопление",
            "base_rate_status": C.BASE_RATE_STATUS_SCENARIO,
            "legal_edition": C.LEGAL_EDITION,
            "methods": {"baseline": "порог по VV", "main": "бустинг по признакам S1"},
            "bounds": bounds,
            "sources": [],
        },
        "assets": {"type": "FeatureCollection", "features": features},
        "candidates": {"type": "FeatureCollection", "features": candidates},
        "sensitivity": [
            {
                "scenario_id": "demo_budget_down",
                "strategy": C.STRATEGY_SELECTIVE,
                "changed_inputs_json": '{"budget_rub": 4000}',
                "decision_cost_rub": "3980.55",
                "covered_expected_loss_rub": "2410000.00",
                "result_status": "demo",
                "interpretation": "Синтетический пример для вёрстки",
            }
        ],
        "_prob": prob,
        "_unc": uncertainty,
        "_bounds": bounds,
    }


def _demo_strategies(payload: dict[str, Any], budget_rub: float | None) -> dict[str, Any]:
    budget = float(budget_rub) if budget_rub is not None else payload["summary"]["budget_rub"]
    catalog = payload["candidates"]["features"]
    broad = [f["properties"]["candidate_id"] for f in catalog]
    broad_cost = sum(f["properties"]["cost_rub"] for f in catalog)

    selective: list[str] = []
    spent = 0.0
    for feature in sorted(catalog, key=lambda f: -f["properties"]["cost_rub"]):
        cost = feature["properties"]["cost_rub"]
        if spent + cost <= budget:
            selective.append(feature["properties"]["candidate_id"])
            spent += cost

    total_el = sum(
        f["properties"]["expected_loss_rub"] or 0.0 for f in payload["assets"]["features"]
    )
    covered = {
        asset
        for f in catalog
        if f["properties"]["candidate_id"] in selective
        for asset in f["properties"]["covered_assets"]
    }
    covered_el = sum(
        f["properties"]["expected_loss_rub"] or 0.0
        for f in payload["assets"]["features"]
        if f["properties"]["asset_id"] in covered
    )

    def row(strategy: str, cost: float, covered_rub: float, feasible: bool) -> dict[str, Any]:
        return {
            "strategy": strategy,
            "data_cost_rub": round(cost, 2),
            "other_cost_rub": 0.0,
            "decision_cost_rub": round(cost, 2),
            "budget_rub": budget,
            "budget_feasible": feasible,
            "covered_expected_loss_rub": round(covered_rub, 2),
            "coverage_share": round(covered_rub / total_el, 4) if total_el else None,
            "residual_uncertainty": None,
            "uncertainty_status": C.UNC_BASELINE if strategy == C.STRATEGY_OPEN else C.UNC_SCENARIO,
            "uncertainty_basis": "демонстрационные данные",
        }

    return {
        "budget_rub": budget,
        "demo": True,
        "plans": {
            C.STRATEGY_OPEN: [],
            C.STRATEGY_BROAD: broad,
            C.STRATEGY_SELECTIVE: selective,
        },
        "positions": [],
        "comparison": [
            row(C.STRATEGY_OPEN, 0.0, 0.0, True),
            row(C.STRATEGY_BROAD, broad_cost, total_el, broad_cost <= budget),
            row(C.STRATEGY_SELECTIVE, spent, covered_el, True),
        ],
    }


def _demo_raster(payload: dict[str, Any], kind: RasterKind) -> tuple[np.ndarray, list[float]]:
    prob = payload["_prob"]
    bounds = payload["_bounds"]
    rgba = np.zeros((*prob.shape, 4), dtype=np.uint8)
    if kind == "s1":
        grey = np.clip(1.0 - 0.8 * prob, 0, 1)
        rgba[..., :3] = (grey * 255).astype(np.uint8)[..., None]
        rgba[..., 3] = 255
    elif kind == "mask":
        water = prob >= payload["summary"]["threshold"]
        rgba[water] = (8, 48, 107, 210)
    else:
        values = prob if kind == "probability" else payload["_unc"]
        stops = _PROB_STOPS if kind == "probability" else _UNC_STOPS
        rgba[..., :3] = _ramp(values, stops)
        rgba[..., 3] = (np.clip(values, 0, 1) * 235).astype(np.uint8)
    return rgba, bounds
