"""Единый контракт проекта FloodValue.

Этот модуль заморожен: на него опираются обе зоны работы (модель и сервис).
Нужна правка — сначала согласование, потом изменение здесь, потом всё остальное.

Здесь и только здесь живут:
  * таблица типов объектов кейса (V и q),
  * коэффициенты и формулы ПП РФ № 840,
  * имена полей всех обязательных выходных файлов,
  * допустимые значения статусов,
  * правила округления и геометрии.

Ничего, что зависит от конкретной модели, чипа или запуска, тут быть не должно.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Final

# ─────────────────────────────────────────────────────────────────────────────
# Источник данных
# ─────────────────────────────────────────────────────────────────────────────

DATASET_NAME: Final = "Sen1Floods11"
DATASET_VERSION: Final = "v1.1"
DATASET_REPO: Final = "https://github.com/cloudtostreet/Sen1Floods11"
DATASET_BUCKET_HTTPS: Final = "https://storage.googleapis.com/sen1floods11"
DATASET_CITATION: Final = (
    "Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. 2020. Sen1Floods11: "
    "a georeferenced dataset to train and test deep learning flood algorithms for "
    "Sentinel-1. CVPR Workshops."
)

#: Слои размеченной части набора, которые мы используем.
LAYER_S1: Final = "S1Hand"
LAYER_LABEL: Final = "LabelHand"
LAYER_JRC: Final = "JRCWaterHand"
LAYER_OTSU: Final = "S1OtsuLabelHand"
HAND_LAYERS: Final = (LAYER_S1, LAYER_LABEL, LAYER_JRC, LAYER_OTSU)

#: Все 11 событий размеченной части и число чипов в каждом (проверено по бакету).
EVENT_CHIP_COUNTS: Final = {
    "USA": 69,
    "India": 68,
    "Paraguay": 67,
    "Ghana": 53,
    "Sri-Lanka": 42,
    "Mekong": 30,
    "Spain": 30,
    "Pakistan": 28,
    "Somalia": 26,
    "Nigeria": 18,
    "Bolivia": 15,
}

# Значения ручной метки Sen1Floods11.
LABEL_INVALID: Final = -1
LABEL_DRY: Final = 0
LABEL_WATER: Final = 1

# Геометрия чипа.
CHIP_SIZE_PX: Final = 512
CHIP_PIXEL_DEG: Final = 9e-05  # ≈ 10 м
CHIP_NOMINAL_AREA_KM2: Final = 26.2144

# ─────────────────────────────────────────────────────────────────────────────
# Растры на выходе
# ─────────────────────────────────────────────────────────────────────────────

PROB_NODATA: Final = -9999.0  # flood_probability.tif, Float32
MASK_WATER: Final = 1  # flood_mask.tif, UInt8
MASK_DRY: Final = 0
MASK_NODATA: Final = 255

CRS_GEO: Final = "EPSG:4326"  # хранение и GeoJSON (порядок lon, lat — CRS84)
CRS_EQUAL_AREA: Final = "EPSG:6933"  # расчёт площадей, применима от 86° ю.ш. до 86° с.ш.

# ─────────────────────────────────────────────────────────────────────────────
# Портфель объектов кейса — значения зафиксированы постановкой
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AssetType:
    """Тип объекта из таблицы кейса. V и q учебные, к реальному имуществу не относятся."""

    asset_class: str
    title_ru: str
    value_rub: int
    vulnerability: float


ASSET_TYPES: Final = (
    AssetType("warehouse", "склад", 10_000_000, 0.25),
    AssetType("substation", "электроподстанция", 20_000_000, 0.40),
    AssetType("road_unit", "участок дороги", 5_000_000, 0.30),
    AssetType("facility", "производственный объект", 8_000_000, 0.20),
    AssetType("pumping_station", "насосная станция", 12_000_000, 0.35),
    AssetType("clinic", "клиника", 30_000_000, 0.15),
    AssetType("school", "школа", 18_000_000, 0.20),
    AssetType("workshop", "мастерская", 6_000_000, 0.45),
    AssetType("telecom_node", "узел связи", 4_000_000, 0.50),
    AssetType("water_intake", "водозабор", 16_000_000, 0.30),
)

ASSET_TYPE_BY_CLASS: Final = {a.asset_class: a for a in ASSET_TYPES}
ASSET_COUNT: Final = 10

assert len(ASSET_TYPES) == ASSET_COUNT

#: Статус оценки объекта. Для partial и no_data поля p/EL/rank остаются ПУСТЫМИ, не нулевыми.
ASSET_STATUS_OK: Final = "ok"
ASSET_STATUS_PARTIAL: Final = "partial"
ASSET_STATUS_NO_DATA: Final = "no_data"
ASSET_STATUSES: Final = (ASSET_STATUS_OK, ASSET_STATUS_PARTIAL, ASSET_STATUS_NO_DATA)

# ─────────────────────────────────────────────────────────────────────────────
# ПП РФ № 840: РП = Б × К × О × П × Т × Р
# ─────────────────────────────────────────────────────────────────────────────

LEGAL_EDITION: Final = "ПП РФ № 840 в ред. от 27.08.2025"
LEGAL_CHECK_DATE: Final = "2026-09-26"
PRICE_FORMULA: Final = "RP = B * K * O * P * T * R"

#: Сценарная ставка БРЕ из материалов кейса. Официальной не является.
BASE_RATE_RUB_KM2_SCENARIO: Final = Decimal("788.64")
BASE_RATE_STATUS_SCENARIO: Final = "scenario"
BASE_RATE_STATUS_OFFICIAL: Final = "official"
BASE_RATE_SOURCE_SCENARIO: Final = (
    "Учебная ставка из постановки кейса КосмоХакатона 2026, контрольная дата 20.09.2026. "
    "Подтверждённый действующий размер БРЕ не установлен."
)

#: О — коэффициент уровня обработки.
PROCESSING_COEF: Final = {
    "L0": Decimal("1"),
    "L1": Decimal("1"),
    "L2": Decimal("1.2"),
}

#: П — коэффициент условий использования.
USAGE_COEF: Final = {
    "internal": Decimal("1"),
    "limited": Decimal("1.2"),
    "unrestricted": Decimal("1.5"),
}

#: Т — коэффициент актуальности. Граница ровно 90 дней трактуется как operational.
FRESHNESS_COEF: Final = {
    "new": Decimal("1.8"),
    "operational": Decimal("1"),
    "archive": Decimal("0.6"),
}
ARCHIVE_AGE_DAYS: Final = 90

SENSOR_OPTICAL: Final = "optical"
SENSOR_SAR: Final = "sar"
SENSOR_TYPES: Final = (SENSOR_OPTICAL, SENSOR_SAR)

DATA_ROLE_EVENT: Final = "event_observation"
DATA_ROLE_CONTEXT: Final = "context"

# Округления учебного расчётного протокола.
AREA_QUANT: Final = Decimal("0.001")  # км²
DISCOUNT_QUANT: Final = Decimal("0.000000001")  # 9 знаков
MONEY_QUANT: Final = Decimal("0.01")  # копейки
ROUNDING: Final = ROUND_HALF_UP

MIN_ORDER_AREA_KM2: Final = 1.0  # проверяется ДО округления


def discount_coef(area_km2: float, resolution_m: float, sensor_type: str) -> Decimal:
    """Коэффициент скидки Р за объём.

    S — площадь корзины в км², r — разрешение в метрах, логарифм натуральный.
    Единицы нормативные: перевод S в м² меняет тариф и является ошибкой.
    """
    if area_km2 <= 0 or resolution_m <= 0:
        raise ValueError("S и r должны быть строго положительны")
    ratio = area_km2 / (resolution_m**2)
    if sensor_type == SENSOR_OPTICAL:
        raw = min(1.0, max(0.2, -0.058 * math.log(ratio) + 1.037))
    elif sensor_type == SENSOR_SAR:
        raw = min(1.0, max(0.7, -0.020 * math.log(ratio) + 1.031))
    else:
        raise ValueError(f"неизвестный тип сенсора: {sensor_type!r}")
    return Decimal(repr(raw)).quantize(DISCOUNT_QUANT, rounding=ROUNDING)


def quantize_money(value: Decimal) -> Decimal:
    return Decimal(value).quantize(MONEY_QUANT, rounding=ROUNDING)


def quantize_area(value: Decimal | float) -> Decimal:
    return Decimal(str(value)).quantize(AREA_QUANT, rounding=ROUNDING)


# ─────────────────────────────────────────────────────────────────────────────
# Стратегии и статусы неопределённости
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_OPEN: Final = "A"  # только открытые данные
STRATEGY_BROAD: Final = "B"  # широкая закупка по заранее объявленному правилу
STRATEGY_SELECTIVE: Final = "C"  # выборочная закупка по приоритетам в пределах бюджета
STRATEGIES: Final = (STRATEGY_OPEN, STRATEGY_BROAD, STRATEGY_SELECTIVE)

#: Статус оценки остаточной неопределённости.
UNC_BASELINE: Final = "baseline"  # исходная неопределённость, стратегия A
UNC_SCENARIO: Final = "scenario"  # эффект предполагаемый, формула раскрыта
UNC_MEASURED: Final = "measured"  # проверено по реальным новым наблюдениям
UNC_NOT_ESTIMATED: Final = "not_estimated"  # числовое поле остаётся пустым
UNCERTAINTY_STATUSES: Final = (UNC_BASELINE, UNC_SCENARIO, UNC_MEASURED, UNC_NOT_ESTIMATED)

# ─────────────────────────────────────────────────────────────────────────────
# Имена и состав обязательных файлов пакета
# ─────────────────────────────────────────────────────────────────────────────

F_PROBABILITY: Final = "flood_probability.tif"
F_MASK: Final = "flood_mask.tif"
F_ASSETS_GEOJSON: Final = "assets.geojson"
F_ASSETS_CSV: Final = "assets.csv"
F_ASSET_LOSS: Final = "asset_loss.csv"
F_CANDIDATES: Final = "candidate_orders.geojson"
F_PROCUREMENT: Final = "procurement_plan.csv"
F_STRATEGY_PLANS: Final = "strategy_plans.json"
F_STRATEGY_COMPARISON: Final = "strategy_comparison.csv"
F_SENSITIVITY: Final = "sensitivity.csv"
F_RUN_METADATA: Final = "run_metadata.json"
F_SOURCE_MANIFEST: Final = "source_manifest.json"

BUNDLE_FILES: Final = (
    F_PROBABILITY,
    F_MASK,
    F_ASSETS_GEOJSON,
    F_ASSETS_CSV,
    F_ASSET_LOSS,
    F_CANDIDATES,
    F_PROCUREMENT,
    F_STRATEGY_PLANS,
    F_STRATEGY_COMPARISON,
    F_SENSITIVITY,
    F_RUN_METADATA,
    F_SOURCE_MANIFEST,
)

COLUMNS_ASSETS: Final = (
    "asset_id",
    "chip_id",
    "asset_class",
    "asset_value_rub",
    "vulnerability_coef",
    "longitude",
    "latitude",
)

COLUMNS_ASSET_LOSS: Final = (
    "asset_id",
    "chip_id",
    "p_flood",
    "expected_loss_rub",
    "uncertainty",
    "rank",
    "status",
)

PROPERTIES_CANDIDATE: Final = (
    "candidate_id",
    "sensor_type",
    "resolution_m",
    "acquisition_type",
    "data_role",
    "observation_at",
    "available_at",
    "processing_level",
    "usage_type",
    "guaranteed_purchase",
    "area_km2",
    "base_rate_rub_km2",
    "base_rate_status",
    "base_rate_source",
)

COLUMNS_PROCUREMENT: Final = (
    "strategy",
    "candidate_id",
    "area_km2",
    "base_rate_rub_km2",
    "base_rate_status",
    "processing_level",
    "usage_type",
    "guaranteed_purchase",
    "processing_coef",
    "usage_coef",
    "freshness_coef",
    "discount_coef",
    "discount_group_id",
    "group_area_km2",
    "unit_price_rub_km2",
    "cost_rub",
    "formula",
    "legal_edition",
)

COLUMNS_STRATEGY_COMPARISON: Final = (
    "strategy",
    "data_cost_rub",
    "other_cost_rub",
    "decision_cost_rub",
    "budget_rub",
    "budget_feasible",
    "covered_expected_loss_rub",
    "coverage_share",
    "residual_uncertainty",
    "uncertainty_status",
    "uncertainty_basis",
)

COLUMNS_SENSITIVITY: Final = (
    "scenario_id",
    "strategy",
    "changed_inputs_json",
    "decision_cost_rub",
    "covered_expected_loss_rub",
    "result_status",
    "interpretation",
)

#: CSV пакета: UTF-8, запятая, десятичная точка, пустая строка для отсутствующих значений.
CSV_ENCODING: Final = "utf-8"
CSV_DELIMITER: Final = ","
CSV_EMPTY: Final = ""

# ─────────────────────────────────────────────────────────────────────────────
# Модель ущерба
# ─────────────────────────────────────────────────────────────────────────────


def expected_loss(p: float, value_rub: int, vulnerability: float) -> float:
    """EL = p × V × q.

    p берётся из исходного вероятностного растра в координате объекта и НЕ зависит
    от бинарной маски: низкая вероятность не обнуляет ожидаемый ущерб.
    """
    return float(p) * float(value_rub) * float(vulnerability)
