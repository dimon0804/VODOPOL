"""Разведочный анализ скачанных чипов Sen1Floods11.

    python -m src.cli.eda
    python -m src.cli.eda --limit 50 --out-dir reports

Проходит по чипам, которые скачаны целиком, собирает статистику каждого и пишет
``reports/chip_stats.csv`` (строка на чип) и ``reports/eda.md`` (таблица по событиям
и короткие выводы). Работает с тем, что уже есть на диске: пока фоновая выгрузка
идёт, отчёт просто строится по доступной части и честно пишет об этом.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from src.contracts import CSV_DELIMITER, CSV_EMPTY, CSV_ENCODING, DATASET_NAME, DATASET_VERSION
from src.contracts import EVENT_CHIP_COUNTS
from src.data import fetch, splits as splits_mod
from src.data.chips import available_chips, chip_stats, iter_chips

COLUMNS: tuple[str, ...] = (
    "chip_id",
    "event",
    "part",
    "pixels",
    "valid_px",
    "valid_share",
    "nodata_share",
    "invalid_label_share",
    "water_share",
    "permanent_share",
    "flood_share",
    "permanent_of_water",
    "vv_min",
    "vv_max",
    "vv_mean",
    "vh_min",
    "vh_max",
    "vh_mean",
)


def _part_by_event() -> dict[str, str]:
    """К какой части сплита относится событие — чтобы EDA и протокол были согласованы."""
    mapping = {event: splits_mod.PART_TRAIN for event in EVENT_CHIP_COUNTS}
    for part, events in (
        (splits_mod.PART_VALID, splits_mod.DEFAULT_VALID_EVENTS),
        (splits_mod.PART_TEST, splits_mod.DEFAULT_TEST_EVENTS),
        (splits_mod.PART_HOLDOUT, splits_mod.DEFAULT_HOLDOUT_EVENTS),
    ):
        for event in events:
            mapping[event] = part
    return mapping


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "нет данных"
    return f"{value * 100:.{digits}f}".replace(".", ",") + " %"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=COLUMNS, delimiter=CSV_DELIMITER, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (CSV_EMPTY if row.get(k) is None else row.get(k)) for k in COLUMNS})


def _by_event(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["event"], []).append(row)
    return grouped


def _report(rows: list[dict[str, Any]], path: Path) -> None:
    """Собирает reports/eda.md — человекочитаемый отчёт на русском."""
    grouped = _by_event(rows)
    parts = _part_by_event()
    summary: list[dict[str, Any]] = []
    for event, items in grouped.items():
        summary.append(
            {
                "event": event,
                "part": parts.get(event, "?"),
                "chips": len(items),
                "total": EVENT_CHIP_COUNTS.get(event, 0),
                "water": _mean(r["water_share"] for r in items),
                "permanent": _mean(r["permanent_share"] for r in items),
                "flood": _mean(r["flood_share"] for r in items),
                "nodata": _mean(r["nodata_share"] for r in items),
                "perm_of_water": _mean(r["permanent_of_water"] for r in items),
            }
        )
    summary.sort(key=lambda s: -(s["flood"] or 0.0))

    lines: list[str] = []
    lines.append(f"# EDA размеченной части {DATASET_NAME} {DATASET_VERSION}")
    lines.append("")
    lines.append(f"Сформировано {date.today().isoformat()} командой `python -m src.cli.eda`.")
    lines.append("")
    covered = sum(s["chips"] for s in summary)
    total_known = sum(EVENT_CHIP_COUNTS.values())
    lines.append(
        f"Разобрано **{covered} чипов из {total_known}** по **{len(summary)} событиям** "
        "(в отчёт попадают только чипы, у которых на диске есть все три обязательных слоя: "
        "`S1Hand`, `LabelHand`, `JRCWaterHand`)."
    )
    lines.append("")
    lines.append("Как читать доли:")
    lines.append("")
    lines.append(
        "* **nodata** — доля пикселей чипа, где радар не дал значения (NaN хотя бы в одном "
        "канале); считается от всех 512x512 пикселей;"
    )
    lines.append(
        "* **вода**, **постоянная вода**, **временное затопление** — доли от ВАЛИДНЫХ "
        "пикселей (метка не -1 и оба канала S1 не NaN), иначе крупная зона NaN искажает "
        "картину;"
    )
    lines.append(
        "* **временное затопление** = `LabelHand == 1 AND JRCWaterHand == 0` — это наш "
        "целевой класс, а не вода вообще;"
    )
    lines.append(
        "* **постоянной от всей воды** — какая часть воды на событии постоянная; чем выше, "
        "тем легче выдать реку или озеро за паводок."
    )
    lines.append("")
    lines.append("## Таблица по событиям")
    lines.append("")
    lines.append(
        "| Событие | Часть сплита | Чипов разобрано | Всего в наборе | Вода | Постоянная вода "
        "| Временное затопление | Постоянной от всей воды | nodata |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in summary:
        lines.append(
            f"| {s['event']} | {s['part']} | {s['chips']} | {s['total']} | "
            f"{_pct(s['water'])} | {_pct(s['permanent'])} | {_pct(s['flood'])} | "
            f"{_pct(s['perm_of_water'])} | {_pct(s['nodata'])} |"
        )
    lines.append("")
    overall = {
        "water": _mean(r["water_share"] for r in rows),
        "permanent": _mean(r["permanent_share"] for r in rows),
        "flood": _mean(r["flood_share"] for r in rows),
        "nodata": _mean(r["nodata_share"] for r in rows),
        "invalid": _mean(r["invalid_label_share"] for r in rows),
    }
    lines.append(
        f"В среднем по разобранным чипам: вода {_pct(overall['water'])}, постоянная вода "
        f"{_pct(overall['permanent'])}, временное затопление {_pct(overall['flood'])}, "
        f"nodata {_pct(overall['nodata'])}, пикселей без метки "
        f"{_pct(overall['invalid'], 2)}."
    )
    lines.append("")

    lines.append("## Баланс частей сплита")
    lines.append("")
    lines.append(
        "Проверка к критерию 6: части, разделённые по событиям, должны быть сопоставимы "
        "не только по числу чипов, но и по редкости целевого класса — иначе подобранный "
        "на validation порог не перенести на test."
    )
    lines.append("")
    lines.append("| Часть | События | Чипов | Вода | Временное затопление |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for part in splits_mod.PARTS:
        part_rows = [r for r in rows if r.get("part") == part]
        if not part_rows:
            continue
        part_events = sorted({r["event"] for r in part_rows})
        lines.append(
            f"| {part} | {', '.join(part_events)} | {len(part_rows)} | "
            f"{_pct(_mean(r['water_share'] for r in part_rows))} | "
            f"{_pct(_mean(r['flood_share'] for r in part_rows))} |"
        )
    lines.append("")

    lines.append("## Что из этого следует")
    lines.append("")
    top = [s for s in summary if s["flood"] is not None][:3]
    bottom = [s for s in summary if s["flood"] is not None][-3:]
    if top:
        names = ", ".join(f"{s['event']} ({_pct(s['flood'])})" for s in top)
        lines.append(
            f"**Где затопления много.** Больше всего временной воды на событиях: {names}. "
            "На них метрики будут выглядеть лучше просто потому, что положительный класс "
            "крупный, — сравнивать методы по одному такому событию нельзя."
        )
        lines.append("")
    if bottom:
        names = ", ".join(f"{s['event']} ({_pct(s['flood'])})" for s in reversed(bottom))
        lines.append(
            f"**Где затопления мало.** Самые бедные на целевой класс события: {names}. "
            "Здесь важен не общий Accuracy, который держится на сухом фоне, а IoU и F1 по "
            "классу воды: на таком дисбалансе они и показывают реальное качество."
        )
        lines.append("")
    perm = sorted(
        (s for s in summary if s["perm_of_water"] is not None),
        key=lambda s: -s["perm_of_water"],
    )[:3]
    if perm:
        names = ", ".join(f"{s['event']} ({_pct(s['perm_of_water'])} воды постоянная)" for s in perm)
        lines.append(
            f"**Где велика доля постоянной воды.** {names}. Именно на этих событиях метод, "
            "который ищет «воду вообще», даст завышенный результат: он будет засчитывать "
            "себе реки и озёра, которые были на месте и до паводка. Поэтому целевой класс "
            "у нас — временное затопление, а слой JRC вычитается до расчёта метрик, а не "
            "после."
        )
        lines.append("")
    noisy = sorted((s for s in summary if s["nodata"] is not None), key=lambda s: -s["nodata"])[:3]
    if noisy and (noisy[0]["nodata"] or 0) > 0.01:
        names = ", ".join(f"{s['event']} ({_pct(s['nodata'])})" for s in noisy)
        lines.append(
            f"**Где много пустых пикселей.** {names}. Такие пиксели исключены и из обучения, "
            "и из знаменателя метрик: «нет данных» — это не «сухо». В выходных растрах им "
            "соответствуют значения nodata, а объекты, попавшие в такую зону, получают "
            "статус `partial` или `no_data` с пустыми полями p/EL/rank."
        )
        lines.append("")
    lines.append(
        "**Про разбиение.** Доли целевого класса заметно различаются от события к событию, "
        "поэтому сравнивать методы по одной случайной нарезке чипов бессмысленно: результат "
        "определяется тем, какое событие куда попало. Мы делим выборку по событиям целиком "
        "(`python -m src.cli.make_splits`), Bolivia держим полностью отложенной."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=fetch.DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--limit", type=int, default=0, help="взять только первые N чипов")
    args = parser.parse_args()

    chip_ids = available_chips(args.root)
    if args.limit:
        chip_ids = chip_ids[: args.limit]
    if not chip_ids:
        print(
            f"в {args.root} нет ни одного полностью скачанного чипа. "
            "Запустите: python -m src.cli.fetch_data",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"чипов к разбору: {len(chip_ids)}", flush=True)
    parts = _part_by_event()
    rows: list[dict[str, Any]] = []
    for index, chip in enumerate(iter_chips(chip_ids, args.root, skip_missing=True), start=1):
        stats = chip_stats(chip)
        stats["part"] = parts.get(chip.event, "?")
        rows.append(stats)
        if index % 25 == 0 or index == len(chip_ids):
            print(f"  обработано {index}/{len(chip_ids)}", flush=True)

    csv_path = args.out_dir / "chip_stats.csv"
    md_path = args.out_dir / "eda.md"
    _write_csv(rows, csv_path)
    _report(rows, md_path)

    events = sorted({row["event"] for row in rows})
    print(f"\nсобытий в выборке: {len(events)} — {', '.join(events)}")
    print(f"построчная статистика: {csv_path}")
    print(f"отчёт: {md_path}")


if __name__ == "__main__":
    main()
