"""Цель запуска, от которой зависят параметры заказа и его цена.

Эксперты на чекпоинте сказали: «роли сделать, побольше индивидуальной адаптации,
так как от цели зависят затраты». Роли с личными кабинетами и разграничением
доступа баллов не дают и в прототипе не нужны, а вот сама мысль верная и
проверяемая: разные люди заказывают съёмку ради разного, и норматив это прямо
учитывает коэффициентами.

Здесь цель — не косметическая подпись, а набор параметров заказа, которые
подставляются в формулу ПП РФ № 840 и меняют итоговую сумму. Ничего не
придумано: все значения берутся из тех же таблиц коэффициентов, что и раньше,
меняется только то, какие именно строки таблицы выбраны.

Разница между целями получается кратной, и это не наша натяжка, а устройство
норматива. Новая съёмка дороже архивной ровно в три раза (Т = 1,8 против 0,6),
потому что за срочность платят. Дежурному по паводку архив бесполезен — вода
уже ушла; оценщику ущерба, наоборот, годится архивная сцена, снятая в пик
разлива, и переплачивать за срочность ему незачем.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Purpose:
    """Одна цель запуска и её параметры заказа."""

    key: str
    title_ru: str
    who: str
    #: Новая, оперативная или архивная съёмка — коэффициент актуальности Т.
    acquisition_type: str
    #: Уровень обработки — коэффициент П.
    processing_level: str
    #: Режим использования — коэффициент О.
    usage_type: str
    #: Гарантированная покупка — влияет на условия поставки.
    guaranteed_purchase: bool
    why: str


#: Три цели, ради которых заказывают съёмку паводка. Набор не выдуман: он
#: собран из того, что перечисляет постановка — оперативное реагирование,
#: оценка ущерба и планирование.
PURPOSES: tuple[Purpose, ...] = (
    Purpose(
        key="response",
        title_ru="оперативное реагирование",
        who="дежурный по паводку",
        acquisition_type="new",
        processing_level="L2",
        usage_type="internal",
        guaranteed_purchase=True,
        why=(
            "Вода стоит сейчас, и решение принимается в течение суток. Архивная "
            "сцена бесполезна: она снята до события или после него. Платим за "
            "срочность — коэффициент актуальности 1,8, самый дорогой в таблице."
        ),
    ),
    Purpose(
        key="damage",
        title_ru="оценка ущерба для компенсаций",
        who="экономист-оценщик",
        acquisition_type="archive",
        processing_level="L2",
        usage_type="internal",
        guaranteed_purchase=True,
        why=(
            "Паводок прошёл, считаются выплаты. Нужна сцена, снятая в пик разлива, "
            "и она к этому моменту уже архивная. Переплачивать за срочность незачем: "
            "коэффициент 0,6 вместо 1,8, то есть втрое дешевле за тот же кадр."
        ),
    ),
    Purpose(
        key="planning",
        title_ru="планирование защитных сооружений",
        who="проектировщик",
        acquisition_type="archive",
        processing_level="L1",
        usage_type="unrestricted",
        guaranteed_purchase=False,
        why=(
            "Данные лягут в проект, который будут читать подрядчики и экспертиза, "
            "поэтому режим использования неограниченный — коэффициент 1,5 вместо "
            "единицы. Зато обработка нужна попроще: проектировщик считает по "
            "геометрии сам, за L2 платить не за что."
        ),
    ),
)

PURPOSE_BY_KEY = {p.key: p for p in PURPOSES}
DEFAULT_PURPOSE = PURPOSES[0].key


def resolve(key: str | None) -> Purpose:
    """Цель по ключу. Неизвестный ключ — ошибка, а не тихий откат к умолчанию."""
    name = (key or DEFAULT_PURPOSE).strip()
    purpose = PURPOSE_BY_KEY.get(name)
    if purpose is None:
        known = ", ".join(p.key for p in PURPOSES)
        raise ValueError(f"неизвестная цель запуска {key!r}; допустимые: {known}")
    return purpose


def as_dict(purpose: Purpose) -> dict[str, Any]:
    """Паспорт цели для метаданных запуска и для панели."""
    return {
        "key": purpose.key,
        "title_ru": purpose.title_ru,
        "who": purpose.who,
        "acquisition_type": purpose.acquisition_type,
        "processing_level": purpose.processing_level,
        "usage_type": purpose.usage_type,
        "guaranteed_purchase": purpose.guaranteed_purchase,
        "why": purpose.why,
    }


def catalog_overrides(purpose: Purpose) -> dict[str, Any]:
    """Поля CatalogConfig, которые задаёт цель.

    Всё остальное в конфигурации каталога цель не трогает: геометрия зон, сенсор
    и разрешение от того, зачем заказывают, не зависят — зависит только то, в
    каком виде и на каких условиях данные берут.
    """
    return {
        "acquisition_type": purpose.acquisition_type,
        "processing_level": purpose.processing_level,
        "usage_type": purpose.usage_type,
        "guaranteed_purchase": purpose.guaranteed_purchase,
    }
