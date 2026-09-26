"""Проверка режима «только Sentinel-1»: запуск на голом радарном снимке.

    python -m src.cli.s1_only --chip Bolivia_103757

Команда существует ровно для того, чтобы утверждение «решение работает только по S1»
можно было проверить, а не принять на слово. Она копирует во временный каталог один
файл S1Hand, читает чип оттуда и считает вероятность затопления. Слоёв LabelHand и
JRCWaterHand рядом нет физически.

Метрики в этом режиме не считаются: без ручной разметки качество измерить нечем, и
код говорит об этом прямо, а не подменяет целевой класс молча.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

from src import contracts as C
from src.data import chips as chips_mod
from src.data import fetch
from src.methods.main_model import MainModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", default="Bolivia_103757")
    parser.add_argument("--model", type=Path, default=Path("models/main"))
    args = parser.parse_args()

    source = fetch.layer_path(args.chip, C.LAYER_S1)
    if not source.exists():
        raise SystemExit(
            f"нет файла {source}. Выгрузите данные: python -m src.cli.fetch_data"
        )

    with tempfile.TemporaryDirectory(prefix="vodopol-s1only-") as tmp:
        root = Path(tmp)
        (root / C.LAYER_S1).mkdir(parents=True)
        shutil.copy2(source, root / C.LAYER_S1 / source.name)
        print(f"во временном каталоге только один слой: {C.LAYER_S1}", flush=True)

        chip = chips_mod.load_chip(args.chip, root)
        model = MainModel.load(args.model)
        prob, uncertainty = model.predict_chip(chip.vv, chip.vh)

        valid = np.isfinite(chip.vv) & np.isfinite(chip.vh)
        water = int((prob >= model.threshold).sum())
        print(f"чип: {chip.chip_id}, событие: {chip.event}")
        print(f"слой меток: {chip.label}, слой постоянной воды: {chip.jrc}")
        print(f"валидных пикселей S1: {int(valid.sum()):,}")
        print(
            f"вероятность посчитана: {prob.shape}, вода при пороге "
            f"{model.threshold}: {water:,} пикселей"
        )
        print(f"средняя неопределённость: {float(uncertainty[valid].mean()):.4f}")
        print(
            "\nметрики не считаются: без ручной разметки качество измерить нечем — "
            "это ограничение режима, а не ошибка"
        )


if __name__ == "__main__":
    main()
