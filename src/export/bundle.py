"""Сборка пакета сдачи одного запуска Водополь.

Запуск — это один чип, один зафиксированный портфель объектов и один объявленный
бюджет. Все двенадцать обязательных файлов такого запуска лежат в ``outputs/<run_id>/``
и образуют согласованный комплект: поменялся бюджет — новый каталог, а не правка
файлов на месте.

Модуль ничего не вычисляет. Вероятности, ущербы, зоны, цены и сравнение стратегий
приходят готовыми, здесь они только раскладываются по файлам через :mod:`src.export.writers`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.contracts import (
    F_ASSET_LOSS,
    F_ASSETS_CSV,
    F_ASSETS_GEOJSON,
    F_CANDIDATES,
    F_MASK,
    F_PROBABILITY,
    F_PROCUREMENT,
    F_RUN_METADATA,
    F_SENSITIVITY,
    F_SOURCE_MANIFEST,
    F_STRATEGY_COMPARISON,
    F_STRATEGY_PLANS,
)
from src.export.writers import (
    BundleWriteError,
    write_asset_loss,
    write_assets,
    write_candidates,
    write_mask_tif,
    write_probability_tif,
    write_procurement_plan,
    write_run_metadata,
    write_sensitivity,
    write_source_manifest,
    write_strategy_comparison,
    write_strategy_plans,
)

__all__ = ["BundlePaths", "assemble", "new_run_id"]

#: Ключи payload, которые обязаны быть заполнены при сборке.
PAYLOAD_KEYS: Sequence[str] = (
    "probability",
    "mask",
    "assets",
    "asset_loss",
    "candidates",
    "procurement",
    "strategy_plans",
    "strategy_comparison",
    "sensitivity",
    "run_metadata",
    "source_manifest",
)

_RUN_ID_SAFE = re.compile(r"[^0-9A-Za-z_.-]+")


@dataclass(frozen=True)
class BundlePaths:
    """Пути ко всем двенадцати файлам пакета внутри каталога запуска."""

    run_dir: Path
    probability: Path
    mask: Path
    assets_geojson: Path
    assets_csv: Path
    asset_loss: Path
    candidates: Path
    procurement: Path
    strategy_plans: Path
    strategy_comparison: Path
    sensitivity: Path
    run_metadata: Path
    source_manifest: Path

    @classmethod
    def for_run(cls, run_dir: Path | str) -> "BundlePaths":
        """Раскладка файлов внутри ``outputs/<run_id>/``. Каталог не создаётся."""
        root = Path(run_dir)
        return cls(
            run_dir=root,
            probability=root / F_PROBABILITY,
            mask=root / F_MASK,
            assets_geojson=root / F_ASSETS_GEOJSON,
            assets_csv=root / F_ASSETS_CSV,
            asset_loss=root / F_ASSET_LOSS,
            candidates=root / F_CANDIDATES,
            procurement=root / F_PROCUREMENT,
            strategy_plans=root / F_STRATEGY_PLANS,
            strategy_comparison=root / F_STRATEGY_COMPARISON,
            sensitivity=root / F_SENSITIVITY,
            run_metadata=root / F_RUN_METADATA,
            source_manifest=root / F_SOURCE_MANIFEST,
        )

    def all_files(self) -> tuple[Path, ...]:
        """Двенадцать обязательных файлов в порядке ``BUNDLE_FILES``."""
        return (
            self.probability,
            self.mask,
            self.assets_geojson,
            self.assets_csv,
            self.asset_loss,
            self.candidates,
            self.procurement,
            self.strategy_plans,
            self.strategy_comparison,
            self.sensitivity,
            self.run_metadata,
            self.source_manifest,
        )


def new_run_id(chip_id: str, budget_rub: int | float | Decimal, at: date | None = None) -> str:
    """Человекочитаемый идентификатор запуска: ``20260926-India_900498-b250000``.

    Детерминирован: одни и те же чип, бюджет и дата дают один и тот же идентификатор.
    Дата передаётся явно, когда нужен воспроизводимый прогон прошлого дня.
    """
    chip = _RUN_ID_SAFE.sub("_", str(chip_id).strip())
    if not chip:
        raise BundleWriteError("chip_id пуст: идентификатор запуска не построить")
    budget = Decimal(str(budget_rub))
    if not budget.is_finite():
        raise BundleWriteError("бюджет должен быть конечным числом")
    if budget < 0:
        raise BundleWriteError("бюджет не может быть отрицательным")
    day = (at or date.today()).strftime("%Y%m%d")
    return f"{day}-{chip}-b{int(budget)}"


def _raster_part(payload: Mapping[str, Any], key: str) -> tuple[Any, Mapping[str, Any]]:
    """Достаёт массив и профиль растра. Профиль можно задать общий на весь payload."""
    part = payload[key]
    if isinstance(part, Mapping):
        data = part.get("data", part.get("array"))
        profile = part.get("profile", payload.get("profile"))
    else:
        data = part
        profile = payload.get("profile")
    if data is None:
        raise BundleWriteError(f"payload[{key!r}]: нет массива растра")
    if profile is None:
        raise BundleWriteError(
            f"payload[{key!r}]: нет профиля исходного чипа (crs, transform, размеры)"
        )
    return data, profile


def assemble(run_dir: Path | str, payload: Mapping[str, Any]) -> BundlePaths:
    """Раскладывает готовые данные запуска по двенадцати файлам пакета.

    ``payload`` — словарь с ключами :data:`PAYLOAD_KEYS`. Растры передаются как
    ``{"data": массив, "profile": профиль чипа}``; общий профиль можно положить
    один раз в ``payload["profile"]``.

    Возвращает :class:`BundlePaths` собранного комплекта.
    """
    missing = [key for key in PAYLOAD_KEYS if payload.get(key) is None]
    if missing:
        raise BundleWriteError(f"в payload не хватает разделов: {missing}")

    paths = BundlePaths.for_run(run_dir)
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    prob_data, prob_profile = _raster_part(payload, "probability")
    mask_data, mask_profile = _raster_part(payload, "mask")
    write_probability_tif(paths.probability, prob_data, prob_profile)
    write_mask_tif(paths.mask, mask_data, mask_profile)

    write_assets(paths.assets_geojson, paths.assets_csv, payload["assets"])
    write_asset_loss(paths.asset_loss, payload["asset_loss"])
    write_candidates(paths.candidates, payload["candidates"])
    write_procurement_plan(paths.procurement, payload["procurement"])
    write_strategy_plans(paths.strategy_plans, payload["strategy_plans"])
    write_strategy_comparison(paths.strategy_comparison, payload["strategy_comparison"])
    write_sensitivity(paths.sensitivity, payload["sensitivity"])
    write_run_metadata(paths.run_metadata, payload["run_metadata"])
    write_source_manifest(paths.source_manifest, payload["source_manifest"])
    return paths
