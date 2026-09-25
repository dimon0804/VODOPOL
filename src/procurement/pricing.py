"""Расчёт платы за заказ данных ДЗЗ по ПП РФ № 840.

Формула нормативная: РП = Б × К × О × П × Т × Р.

  Б — стоимость одной базовой расчётной единицы, руб./км² (``base_rate_rub_km2``);
  К — число БРЕ, 1 БРЕ = 1 км², то есть площадь зоны (``area_km2``);
  О — коэффициент уровня обработки (``processing_coef``);
  П — коэффициент условий использования (``usage_coef``);
  Т — коэффициент актуальности (``freshness_coef``);
  Р — коэффициент скидки за объём (``discount_coef``).

Главное содержательное правило модуля: **скидка Р считается не по площади одной зоны,
а по суммарной площади S всей группы в корзине** (плюс площадь ранее учтённых заказов
той же группы в пределах календарного года). Поэтому цену корзины нельзя получить
сложением заранее посчитанных цен одиночных зон: при любом изменении состава корзины
группа пересчитывается целиком. Точка входа — :func:`price_basket`.

Все деньги и коэффициенты считаются на ``Decimal`` с ROUND_HALF_UP; константы,
коэффициенты и правила округления берутся только из ``src.contracts``.

Контрольный пример постановки (Б = 788,64; К = 1,000 км²; оптика r = 1 м; S = 1,000 км²;
L2; internal; new) даёт 1 703,46 руб./км² и 1 703,46 руб. за позицию; при operational —
946,37; при archive — 567,82.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable, Mapping, Sequence

from src.contracts import (
    ARCHIVE_AGE_DAYS,
    FRESHNESS_COEF,
    LEGAL_EDITION,
    MIN_ORDER_AREA_KM2,
    PRICE_FORMULA,
    PROCESSING_COEF,
    SENSOR_TYPES,
    USAGE_COEF,
    CSV_EMPTY,
    COLUMNS_PROCUREMENT,
    discount_coef,
    quantize_area,
    quantize_money,
)

__all__ = [
    "OrderSpec",
    "PricedPosition",
    "ACQUISITION_TYPES",
    "PROCESSING_LEVELS",
    "USAGE_TYPES",
    "discount_group_key",
    "freshness_from_age",
    "price_basket",
    "basket_total",
    "position_row",
    "explain",
]

#: Т — способ получения данных. Соответствует ключам FRESHNESS_COEF.
ACQUISITION_NEW = "new"
ACQUISITION_OPERATIONAL = "operational"
ACQUISITION_ARCHIVE = "archive"
ACQUISITION_TYPES = (ACQUISITION_NEW, ACQUISITION_OPERATIONAL, ACQUISITION_ARCHIVE)

#: О — допустимые уровни обработки закупаемого продукта.
PROCESSING_LEVELS = tuple(PROCESSING_COEF)

#: П — допустимые условия использования.
USAGE_TYPES = tuple(USAGE_COEF)

#: Календарный год по умолчанию: площадь для скидки копится в пределах года.
DEFAULT_YEAR = 2026


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательное
# ─────────────────────────────────────────────────────────────────────────────


def _to_decimal(value: object, name: str) -> Decimal:
    """Аккуратный перевод в Decimal: float идёт через str, чтобы не тащить двоичный хвост."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name}: ожидалось число, получено {value!r}")
    if isinstance(value, (int, float, str)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            raise ValueError(f"{name}: не удалось прочитать число из {value!r}") from None
    raise ValueError(f"{name}: ожидалось число, получено {type(value).__name__}")


def _format_resolution(resolution_m: Decimal) -> str:
    """Разрешение в идентификаторе группы: без лишних нулей, 1.0 и 1 — одна группа."""
    normalized = resolution_m.normalize()
    if normalized == normalized.to_integral_value():
        normalized = normalized.to_integral_value()
    text = format(normalized, "f")
    return text


def _format_number(value: Decimal) -> str:
    """Число для расшифровки: без экспоненты и без хвостовых нулей."""
    text = format(value.normalize(), "f")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Параметры одной кандидатной зоны
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OrderSpec:
    """Параметры одной кандидатной зоны заказа — вход расчёта цены.

    Одна зона — одна оплачиваемая позиция: полигон без внутренних вырезов площадью
    не менее ``MIN_ORDER_AREA_KM2`` км² **до** округления, оплачивается по контуру.
    Пересечение зон цену не уменьшает.

    Поля:
      candidate_id        — идентификатор зоны;
      sensor_type         — 'optical' | 'sar', задаёт формулу скидки;
      resolution_m        — разрешение r в метрах, строго положительное;
      acquisition_type    — 'new' | 'operational' | 'archive', даёт Т;
      processing_level    — 'L0' | 'L1' | 'L2', даёт О;
      usage_type          — 'internal' | 'limited' | 'unrestricted', даёт П;
      guaranteed_purchase — гарантированная покупка, обязательна для новой съёмки;
      area_km2            — площадь зоны К в км²;
      base_rate_rub_km2   — ставка Б, руб./км²;
      base_rate_status    — 'scenario' или 'official', в пакет уходит как есть;
      discount_group_id   — явный идентификатор группы скидки; None — вычисляется.
    """

    candidate_id: str
    sensor_type: str
    resolution_m: Decimal
    acquisition_type: str
    processing_level: str
    usage_type: str
    guaranteed_purchase: bool
    area_km2: Decimal
    base_rate_rub_km2: Decimal
    base_rate_status: str
    discount_group_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id: требуется непустой строковый идентификатор зоны")

        if self.sensor_type not in SENSOR_TYPES:
            raise ValueError(
                f"sensor_type: {self.sensor_type!r} — неизвестный тип сенсора, "
                f"допустимы {list(SENSOR_TYPES)}"
            )

        if self.acquisition_type not in ACQUISITION_TYPES:
            raise ValueError(
                f"acquisition_type: {self.acquisition_type!r} — неизвестный способ получения "
                f"данных, допустимы {list(ACQUISITION_TYPES)}"
            )

        if self.processing_level not in PROCESSING_COEF:
            raise ValueError(
                f"processing_level: {self.processing_level!r} — неизвестный уровень обработки, "
                f"допустимы {list(PROCESSING_LEVELS)}"
            )

        if self.usage_type not in USAGE_COEF:
            raise ValueError(
                f"usage_type: {self.usage_type!r} — неизвестные условия использования, "
                f"допустимы {list(USAGE_TYPES)}"
            )

        if not isinstance(self.guaranteed_purchase, bool):
            raise ValueError("guaranteed_purchase: ожидалось True или False")

        if self.acquisition_type == ACQUISITION_NEW and not self.guaranteed_purchase:
            raise ValueError(
                "guaranteed_purchase: коэффициент актуальности 1,8 относится к новой съёмке "
                "с гарантированной покупкой; для acquisition_type='new' нужен True"
            )

        self.resolution_m = _to_decimal(self.resolution_m, "resolution_m")
        if self.resolution_m <= 0:
            raise ValueError(
                f"resolution_m: разрешение должно быть строго положительным, "
                f"получено {self.resolution_m}"
            )

        self.area_km2 = _to_decimal(self.area_km2, "area_km2")
        if self.area_km2 <= 0:
            raise ValueError(
                f"area_km2: площадь зоны должна быть строго положительной, "
                f"получено {self.area_km2}"
            )
        if self.area_km2 < Decimal(str(MIN_ORDER_AREA_KM2)):
            raise ValueError(
                f"area_km2: зона заказа меньше {MIN_ORDER_AREA_KM2} км² до округления "
                f"({self.area_km2}) — такая позиция не заказывается"
            )

        self.base_rate_rub_km2 = _to_decimal(self.base_rate_rub_km2, "base_rate_rub_km2")
        if self.base_rate_rub_km2 <= 0:
            raise ValueError(
                f"base_rate_rub_km2: ставка Б должна быть строго положительной, "
                f"получено {self.base_rate_rub_km2}"
            )

        if not isinstance(self.base_rate_status, str) or not self.base_rate_status.strip():
            raise ValueError("base_rate_status: требуется непустая строка ('scenario'/'official')")

        if self.discount_group_id is not None and (
            not isinstance(self.discount_group_id, str) or not self.discount_group_id.strip()
        ):
            raise ValueError("discount_group_id: требуется непустая строка либо None")

    # Коэффициенты позиции — читаются напрямую из контракта.

    @property
    def processing_coef(self) -> Decimal:
        """О — коэффициент уровня обработки."""
        return PROCESSING_COEF[self.processing_level]

    @property
    def usage_coef(self) -> Decimal:
        """П — коэффициент условий использования."""
        return USAGE_COEF[self.usage_type]

    @property
    def freshness_coef(self) -> Decimal:
        """Т — коэффициент актуальности."""
        return FRESHNESS_COEF[self.acquisition_type]

    @property
    def rounded_area_km2(self) -> Decimal:
        """Площадь зоны, округлённая до 0,001 км² — она же К в формуле."""
        return quantize_area(self.area_km2)


# ─────────────────────────────────────────────────────────────────────────────
# Группировка для скидки
# ─────────────────────────────────────────────────────────────────────────────


def discount_group_key(spec: OrderSpec, year: int = DEFAULT_YEAR) -> str:
    """Ключ группы, в пределах которой копится площадь S для скидки Р.

    Группировка учебного протокола — тип данных + разрешение + календарный год.
    Если у зоны задан явный ``discount_group_id``, он и есть ключ: это позволяет
    объединять или разводить группы вручную, не ломая расчёт.

    Год по умолчанию приходит параметром расчёта (:func:`price_basket`), потому что
    площадь копится в пределах календарного года, а не на всё время.
    """
    if spec.discount_group_id is not None:
        return spec.discount_group_id
    if not isinstance(year, int) or isinstance(year, bool):
        raise ValueError(f"year: ожидался календарный год целым числом, получено {year!r}")
    return f"{spec.sensor_type}_r{_format_resolution(spec.resolution_m)}_{year}"


def freshness_from_age(days: int) -> str:
    """Тип актуальности имеющихся данных по их давности в днях.

    Строго больше ``ARCHIVE_AGE_DAYS`` (90) — 'archive' (Т = 0,6), иначе 'operational'
    (Т = 1). Ровно на 90 днях применяется 1 — граница принадлежит оперативным данным.

    Для новой съёмки тип не выводится из давности, а задаётся явно как 'new'.
    """
    if isinstance(days, bool) or not isinstance(days, int):
        raise ValueError(f"days: ожидалось целое число дней, получено {days!r}")
    if days < 0:
        raise ValueError(f"days: давность не может быть отрицательной, получено {days}")
    return ACQUISITION_ARCHIVE if days > ARCHIVE_AGE_DAYS else ACQUISITION_OPERATIONAL


# ─────────────────────────────────────────────────────────────────────────────
# Результат расчёта одной позиции
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PricedPosition:
    """Посчитанная позиция закупки — ровно строка ``procurement_plan.csv``.

    Состав и имена полей совпадают с ``COLUMNS_PROCUREMENT``: наружу отдаются все
    промежуточные множители, чтобы цену можно было пересчитать руками по формуле
    ``RP = B * K * O * P * T * R``, не заглядывая в код.

    ``group_area_km2`` — та самая S, по которой считалась скидка: площадь всей группы
    в корзине плюс ранее учтённые заказы этой группы. Она, как правило, больше
    ``area_km2`` отдельной зоны — это не ошибка, а смысл объёмной скидки.
    """

    strategy: str | None
    candidate_id: str
    area_km2: Decimal
    base_rate_rub_km2: Decimal
    base_rate_status: str
    processing_level: str
    usage_type: str
    guaranteed_purchase: bool
    processing_coef: Decimal
    usage_coef: Decimal
    freshness_coef: Decimal
    discount_coef: Decimal
    discount_group_id: str
    group_area_km2: Decimal
    unit_price_rub_km2: Decimal
    cost_rub: Decimal
    formula: str = PRICE_FORMULA
    legal_edition: str = LEGAL_EDITION


def position_row(position: PricedPosition) -> dict[str, object]:
    """Строка для ``procurement_plan.csv`` в порядке ``COLUMNS_PROCUREMENT``.

    ``None`` превращается в пустую строку по правилу пакета: пустое значение, не ноль.
    """
    row: dict[str, object] = {}
    for column in COLUMNS_PROCUREMENT:
        value = getattr(position, column)
        row[column] = CSV_EMPTY if value is None else value
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Расчёт корзины
# ─────────────────────────────────────────────────────────────────────────────


def price_basket(
    specs: Sequence[OrderSpec],
    prior_area_by_group: Mapping[str, Decimal] | None = None,
    year: int = DEFAULT_YEAR,
    strategy: str | None = None,
) -> list[PricedPosition]:
    """Посчитать цены всех позиций корзины с пересчётом скидки по группам целиком.

    Порядок расчёта:

      1. Позиции раскладываются по группам скидки (:func:`discount_group_key`).
      2. Для каждой группы S = сумма площадей её позиций в корзине + площадь ранее
         учтённых заказов этой группы из ``prior_area_by_group`` (по умолчанию 0).
      3. Р = ``discount_coef(S, r, sensor)`` — один на всю группу, до 9 знаков.
      4. Для каждой позиции цена за км² = Б × О × П × Т × Р, округление до копеек.
      5. Стоимость позиции = цена за км² × К (площадь зоны), округление до копеек.

    Именно поэтому нельзя складывать заранее посчитанные цены одиночных зон: добавление
    зоны в группу меняет S, а значит и Р, а значит и цену всех остальных позиций группы.

    ``prior_area_by_group`` — площадь, уже набранная по группе в текущем календарном году
    вне этой корзины. ``year`` участвует только в ключе группы. ``strategy`` — пометка
    сценария (A/B/C), она уходит в CSV и на расчёт не влияет.

    Возвращает список ``PricedPosition`` в исходном порядке ``specs``.
    """
    if specs is None:
        raise ValueError("specs: ожидался список позиций, получено None")
    specs = list(specs)
    for index, spec in enumerate(specs):
        if not isinstance(spec, OrderSpec):
            raise ValueError(f"specs[{index}]: ожидался OrderSpec, получено {type(spec).__name__}")

    seen_ids: set[str] = set()
    for spec in specs:
        if spec.candidate_id in seen_ids:
            raise ValueError(
                f"candidate_id: {spec.candidate_id!r} встречается в корзине дважды — "
                "каждая зона это отдельная позиция с собственным идентификатором"
            )
        seen_ids.add(spec.candidate_id)

    # 1. Раскладка по группам с сохранением порядка появления.
    group_keys: list[str] = [discount_group_key(spec, year) for spec in specs]
    members: dict[str, list[OrderSpec]] = {}
    for key, spec in zip(group_keys, specs):
        members.setdefault(key, []).append(spec)

    # Внутри группы тип сенсора и разрешение обязаны совпадать, иначе формула скидки
    # неопределена. Это может случиться только при ручном discount_group_id.
    for key, group in members.items():
        sensors = {spec.sensor_type for spec in group}
        resolutions = {spec.resolution_m for spec in group}
        if len(sensors) > 1 or len(resolutions) > 1:
            raise ValueError(
                f"discount_group_id={key!r}: в одной группе скидки оказались разные тип сенсора "
                f"или разрешение ({sorted(sensors)}, {sorted(resolutions)}) — "
                "скидка для такой группы не определена"
            )

    # 2. Площадь ранее учтённых заказов по группе.
    prior: dict[str, Decimal] = {}
    if prior_area_by_group:
        for key, value in prior_area_by_group.items():
            area = _to_decimal(value, f"prior_area_by_group[{key!r}]")
            if area < 0:
                raise ValueError(
                    f"prior_area_by_group[{key!r}]: ранее учтённая площадь не может быть "
                    f"отрицательной, получено {area}"
                )
            prior[key] = quantize_area(area)

    # 3. S и Р на группу.
    group_area: dict[str, Decimal] = {}
    group_discount: dict[str, Decimal] = {}
    for key, group in members.items():
        basket_area = sum((spec.rounded_area_km2 for spec in group), Decimal("0"))
        total_area = quantize_area(basket_area + prior.get(key, Decimal("0")))
        sample = group[0]
        group_area[key] = total_area
        group_discount[key] = discount_coef(
            float(total_area), float(sample.resolution_m), sample.sensor_type
        )

    # 4-5. Цена за км² и стоимость позиции.
    positions: list[PricedPosition] = []
    for key, spec in zip(group_keys, specs):
        discount = group_discount[key]
        area = spec.rounded_area_km2
        raw_unit_price = (
            spec.base_rate_rub_km2
            * spec.processing_coef
            * spec.usage_coef
            * spec.freshness_coef
            * discount
        )
        unit_price = quantize_money(raw_unit_price)
        cost = quantize_money(unit_price * area)
        positions.append(
            PricedPosition(
                strategy=strategy,
                candidate_id=spec.candidate_id,
                area_km2=area,
                base_rate_rub_km2=spec.base_rate_rub_km2,
                base_rate_status=spec.base_rate_status,
                processing_level=spec.processing_level,
                usage_type=spec.usage_type,
                guaranteed_purchase=spec.guaranteed_purchase,
                processing_coef=spec.processing_coef,
                usage_coef=spec.usage_coef,
                freshness_coef=spec.freshness_coef,
                discount_coef=discount,
                discount_group_id=key,
                group_area_km2=group_area[key],
                unit_price_rub_km2=unit_price,
                cost_rub=cost,
            )
        )
    return positions


def basket_total(positions: Iterable[PricedPosition]) -> Decimal:
    """Итог корзины — сумма стоимостей позиций, без пересчёта с нуля.

    Протокол прямо требует складывать уже округлённые цены позиций: так итог совпадает
    со строками ``procurement_plan.csv``, и жюри сходится копейка в копейку.
    """
    total = Decimal("0")
    for position in positions:
        if not isinstance(position, PricedPosition):
            raise ValueError(
                f"positions: ожидался PricedPosition, получено {type(position).__name__}"
            )
        total += position.cost_rub
    return quantize_money(total)


# ─────────────────────────────────────────────────────────────────────────────
# Расшифровка для интерфейса и защиты
# ─────────────────────────────────────────────────────────────────────────────


def explain(position: PricedPosition) -> str:
    """Человекочитаемая расшифровка расчёта позиции: формула, числа, результат.

    Текст рассчитан на оператора и на защиту: по нему цену можно пересчитать на
    калькуляторе, не открывая код.
    """
    lines = [
        f"Позиция {position.candidate_id}"
        + (f" (стратегия {position.strategy})" if position.strategy else ""),
        f"Формула РП = Б × К × О × П × Т × Р ({PRICE_FORMULA}), {position.legal_edition}.",
        f"Б — ставка {_format_number(position.base_rate_rub_km2)} руб./км² "
        f"(статус: {position.base_rate_status}).",
        f"К — площадь зоны {position.area_km2} км².",
        f"О — уровень обработки {position.processing_level}: "
        f"{_format_number(position.processing_coef)}.",
        f"П — условия использования {position.usage_type}: "
        f"{_format_number(position.usage_coef)}.",
        f"Т — актуальность: {_format_number(position.freshness_coef)}"
        + (", гарантированная покупка" if position.guaranteed_purchase else "")
        + ".",
        f"Р — скидка за объём {_format_number(position.discount_coef)} при S = "
        f"{_format_number(position.group_area_km2)} км² по группе "
        f"{position.discount_group_id} (площадь всей группы в корзине, а не одной зоны).",
        f"Цена за км² = {_format_number(position.base_rate_rub_km2)} × "
        f"{_format_number(position.processing_coef)} × "
        f"{_format_number(position.usage_coef)} × "
        f"{_format_number(position.freshness_coef)} × "
        f"{_format_number(position.discount_coef)} = "
        f"{position.unit_price_rub_km2} руб./км².",
        f"Стоимость позиции = {position.unit_price_rub_km2} × "
        f"{position.area_km2} = {position.cost_rub} руб.",
    ]
    return "\n".join(lines)
