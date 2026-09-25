"""Построение собственного сплита по событиям и проверка его на утечку.

    python -m src.cli.make_splits
    python -m src.cli.make_splits --seed 7 --out-dir splits_alt
    python -m src.cli.make_splits --valid-events Spain Nigeria --test-events Mekong Somalia

Команда печатает состав частей, результат проверки утечки и, для сравнения, те же
числа по официальному split-у авторов набора — именно он даёт повод сделать свой.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.contracts import EVENT_CHIP_COUNTS
from src.data import fetch, splits as splits_mod
from src.data.chips import available_chips, event_of


def _print_part(part: str, splits: dict, root: Path) -> None:
    chips = splits["chips"][part]
    total = len(chips)
    share = 100.0 * total / max(1, sum(splits["counts"].values()))
    print(f"\n{part}: {total} чипов ({share:.1f} % набора)")
    if not chips:
        print("  (пусто)")
        return
    by_event: dict[str, int] = {}
    for chip_id in chips:
        event = event_of(chip_id)
        by_event[event] = by_event.get(event, 0) + 1
    on_disk = set(available_chips(root))
    for event in sorted(by_event, key=lambda e: -by_event[e]):
        expected = EVENT_CHIP_COUNTS.get(event, 0)
        ready = sum(1 for c in chips if event_of(c) == event and c in on_disk)
        print(f"  {event:<12} {by_event[event]:>3} чипов (в бакете {expected}, скачано {ready})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=fetch.DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=splits_mod.DEFAULT_SPLIT_DIR)
    parser.add_argument("--seed", type=int, default=splits_mod.SPLIT_SEED)
    parser.add_argument("--valid-events", nargs="*", default=list(splits_mod.DEFAULT_VALID_EVENTS))
    parser.add_argument("--test-events", nargs="*", default=list(splits_mod.DEFAULT_TEST_EVENTS))
    parser.add_argument(
        "--holdout-events", nargs="*", default=list(splits_mod.DEFAULT_HOLDOUT_EVENTS)
    )
    parser.add_argument(
        "--only-downloaded",
        action="store_true",
        help="строить сплит только по скачанным чипам, а не по всему перечню набора",
    )
    args = parser.parse_args()

    if args.only_downloaded:
        chip_ids = available_chips(args.root)
        source = "скачанные чипы"
    else:
        chip_ids = splits_mod.known_chip_ids(args.root)
        source = "полный перечень набора (авторские split-файлы + скачанное)"
    if not chip_ids:
        print(
            "чипов не найдено. Сначала выгрузите данные: python -m src.cli.fetch_data",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"источник списка чипов: {source} — {len(chip_ids)} шт.")
    print(f"правило: {splits_mod.SPLIT_RULE}")
    print(f"seed: {args.seed}")

    splits = splits_mod.split_by_event(
        chip_ids,
        seed=args.seed,
        valid_events=args.valid_events,
        test_events=args.test_events,
        holdout_events=args.holdout_events,
    )
    for part in splits_mod.PARTS:
        _print_part(part, splits, args.root)

    out = splits_mod.save_splits(splits, args.out_dir)
    print(f"\nсохранено в {out}/: " + ", ".join(f"{p}.csv" for p in splits_mod.PARTS))
    print(f"манифест: {out / splits_mod.MANIFEST_NAME}")

    problems = splits_mod.check_no_leakage(splits)
    print("\nпроверка утечки нашего сплита:")
    if problems:
        for item in problems:
            print(f"  НАРУШЕНИЕ: {item}")
    else:
        print("  нарушений нет: общих событий между частями нет, дубликатов нет, пустых частей нет")

    print("\nдля сравнения — официальный split авторов набора:")
    author = splits_mod.author_split_overlap(args.root)
    if not author.get("available"):
        print(f"  не проверен: {author.get('reason')}")
    else:
        for part, count in author["chips"].items():
            print(f"  {part:<11} {count:>3} чипов, событий: {author['event_counts'][part]}")
        for pair, info in author["pairwise"].items():
            left, right = pair.split("|")
            print(
                f"  общих событий {left} и {right}: {info['shared_events']} "
                f"(общих чипов {info['shared_chips']})"
            )
        print(
            f"  событий, общих сразу для train/valid/test: "
            f"{author['events_shared_by_all_three_count']} — "
            f"{', '.join(author['events_shared_by_all_three']) or 'нет'}"
        )
        if author["leakage"]:
            print(
                "  вывод: части авторов нарезаны случайно по чипам внутри одних и тех же "
                "событий, соседние фрагменты одной сцены попадают в разные части — "
                "это утечка, поэтому используем свой сплит (ADR-0001)"
            )

    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
