"""Выгрузка размеченной части Sen1Floods11.

    python -m src.cli.fetch_data                 # все 446 чипов, четыре слоя
    python -m src.cli.fetch_data --limit 20      # быстрая проверка на подвыборке
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.contracts import HAND_LAYERS
from src.data import fetch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=fetch.DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=0, help="взять только первые N чипов")
    parser.add_argument("--layers", nargs="*", default=list(HAND_LAYERS))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    started = time.time()
    print("список чипов из бакета…", flush=True)
    chips = fetch.chip_ids()
    if args.limit:
        chips = chips[: args.limit]
    print(f"чипов: {len(chips)}, слоёв: {len(args.layers)}", flush=True)

    fetch.fetch_author_splits(args.root)
    fetch.fetch_metadata(args.root)
    print("split-файлы авторов и метаданные набора на месте", flush=True)

    done = fetch.fetch_layers(chips, args.layers, args.root, args.workers)
    total_mb = sum(p.stat().st_size for p in args.root.rglob("*.tif")) / (1 << 20)
    print(f"готово за {time.time() - started:.0f} с: {done}, {total_mb:.0f} МБ", flush=True)


if __name__ == "__main__":
    main()
