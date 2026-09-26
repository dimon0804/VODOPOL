"""Тот же расчёт на снимке из любого источника, а не только из Sen1Floods11.

Зачем. На чекпоинте эксперты сказали прямо: источники у вас фиксированные,
нужно уметь брать те же территории из других источников. Замечание верное.
Обучающий набор Sen1Floods11 фиксирован по определению — на нём модель училась,
и менять его задним числом нельзя. А вот **применение** привязано к набору без
всякой на то причины: на вход основному методу идут два канала радарного снимка
в децибелах, и откуда взялся файл с этими каналами, модели безразлично.

Этот модуль снимает привязку. Он принимает произвольный GeoTIFF с VV и VH —
выгруженный из Copernicus Data Space, из ASF, из архива оператора, нарезанный
собственным скриптом — и приводит его к той форме, которую ждёт остальной
конвейер: два канала float32, NaN на месте пропусков, квадратный кадр 512×512.

Чего он сознательно НЕ делает:

* не калибрует сырые значения. Если в файле не децибелы, а амплитуда или
  линейная интенсивность, числа будут другого порядка, и модель ответит ерундой.
  Поэтому диапазон проверяется и при подозрении выдаётся внятная ошибка, а не
  тихий мусор на выходе;
* не перепроецирует и не ресемплирует ради «лишь бы заработало». Шаг сетки
  проверяется, и если он далёк от десяти метров, это тоже ошибка: признаки
  контекста считаются окнами в пикселях, и на другом масштабе окно 51 пиксель
  означает совсем другое расстояние на земле;
* не притворяется, что качество на чужом источнике измерено. Ручной разметки к
  такому снимку нет, метрики посчитать не на чем, и в паспорт запуска попадает
  честная пометка об этом.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from src.data.chips import Chip, ChipError, CHIP_SIZE_PX

#: Ожидаемый шаг сетки в градусах и допуск. Признаки контекста считаются окнами
#: в пикселях, поэтому масштаб пикселя — часть контракта модели, а не мелочь.
EXPECTED_STEP_DEG = 9e-05
STEP_TOLERANCE = 0.5  # допускаем двукратное отличие в любую сторону

#: Правдоподобный диапазон обратного рассеяния Sentinel-1 в децибелах.
#: Значения сильно за его пределами означают, что файл не в дБ.
DB_RANGE = (-60.0, 20.0)


def _looks_like_db(values: np.ndarray) -> bool:
    """Похожи ли числа на обратное рассеяние в децибелах.

    Различать приходится с линейной интенсивностью, и главный признак —
    знак. Обратное рассеяние Sentinel-1 над сушей практически всегда
    отрицательное: на реальном чипе India_900498 медиана −16,7 дБ, а
    отрицательных значений ровно сто процентов. Линейная интенсивность, наоборот,
    неотрицательна по определению и держится около сотых долей единицы.

    Проверяем поэтому и диапазон, и знак: диапазон [−60; 20] сам по себе пропускает
    линейные значения вроде 0,05, и одной только границы не хватает.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    low = float(np.percentile(finite, 1))
    high = float(np.percentile(finite, 99))
    if not (DB_RANGE[0] <= low and high <= DB_RANGE[1]):
        return False
    return float(np.median(finite)) < 0.0


def load_external_chip(
    path: Path,
    chip_id: str,
    *,
    vv_band: int = 1,
    vh_band: int = 2,
    nodata: float | None = None,
) -> Chip:
    """Читает произвольный радарный снимок как чип для применения модели.

    Возвращает тот же ``Chip``, что и штатная загрузка, но без ``label`` и без
    ``jrc``: ручной разметки к чужому снимку нет, и подставлять на её место
    что-либо — значит выдумывать истину. Все функции, которым метка нужна,
    честно сообщат, что её нет.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise ChipError(f"снимок {path} не найден или пуст")

    with rasterio.open(path) as src:
        if src.count < max(vv_band, vh_band):
            raise ChipError(
                f"в файле {path.name} {src.count} канал(ов), а запрошены "
                f"{vv_band} и {vh_band}. Укажите номера каналов явно, если порядок другой."
            )
        vv = src.read(vv_band).astype("float32")
        vh = src.read(vh_band).astype("float32")
        transform = src.transform
        crs = src.crs
        profile = dict(src.profile)
        file_nodata = src.nodata

    # Пропуски приводим к NaN: дальше по конвейеру именно NaN означает «нет данных».
    fill = nodata if nodata is not None else file_nodata
    if fill is not None and np.isfinite(fill):
        vv = np.where(np.isclose(vv, fill), np.nan, vv)
        vh = np.where(np.isclose(vh, fill), np.nan, vh)

    height, width = vv.shape
    if (height, width) != (CHIP_SIZE_PX, CHIP_SIZE_PX):
        raise ChipError(
            f"снимок {path.name} имеет размер {width}×{height}, а конвейер работает с "
            f"кадром {CHIP_SIZE_PX}×{CHIP_SIZE_PX}. Нарежьте сцену на кадры такого "
            "размера — это делается одной командой gdal_retile или rasterio."
        )

    step = abs(float(transform.a))
    low = EXPECTED_STEP_DEG * STEP_TOLERANCE
    high = EXPECTED_STEP_DEG / STEP_TOLERANCE
    if crs is not None and crs.is_geographic and not (low <= step <= high):
        raise ChipError(
            f"шаг сетки {step:.2e}° далёк от ожидаемых {EXPECTED_STEP_DEG:.0e}° "
            "(примерно 10 м). Признаки контекста считаются окнами в пикселях, и на "
            "другом масштабе окно означает другое расстояние на земле — результат "
            "был бы несопоставим с тем, на чём модель училась."
        )

    if not _looks_like_db(np.concatenate([vv.ravel(), vh.ravel()])):
        raise ChipError(
            f"значения в {path.name} не похожи на децибелы (ожидается примерно от "
            f"{DB_RANGE[0]:g} до {DB_RANGE[1]:g} дБ). Скорее всего, это амплитуда или "
            "линейная интенсивность: переведите в дБ как 10·log10(интенсивность) "
            "перед запуском, иначе модель получит числа другого порядка."
        )

    return Chip(
        chip_id=chip_id,
        event=chip_id.rsplit("_", 1)[0] if "_" in chip_id else chip_id,
        vv=vv,
        vh=vh,
        label=None,
        jrc=None,
        transform=transform,
        crs=crs,
        profile=profile,
    )


def describe(path: Path) -> dict:
    """Паспорт чужого снимка для манифеста источников.

    В манифест попадает то, что действительно известно о файле, и ничего сверх:
    выдумывать оператора, дату съёмки или лицензию за пользователя нельзя, их
    он указывает сам через файл дополнительных источников.
    """
    path = Path(path)
    with rasterio.open(path) as src:
        return {
            "file": path.as_posix(),
            "bands": src.count,
            "size": [src.width, src.height],
            "crs": str(src.crs) if src.crs else "",
            "pixel_deg": abs(float(src.transform.a)),
            "dtype": str(src.dtypes[0]) if src.dtypes else "",
            "size_bytes": path.stat().st_size,
        }
