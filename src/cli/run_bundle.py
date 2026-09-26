"""Полный прогон одного запуска: от чипа Sentinel-1 до пакета сдачи.

    python -m src.cli.run_bundle --chip India_900498 --budget 250000
    python -m src.cli.run_bundle --chip Bolivia_103757 --budget 250000   # другой чип, без подгонки

Последовательность ровно та, которую кейс требует показать на защите:
загрузка S1 → вероятность и маска → десять объектов и ущерб → генерация и отбор зон →
A/B/C при бюджете → чувствительность → пакет файлов.

Модель здесь только применяется. Ни обучения, ни подбора порога тут нет: и то и
другое делается заранее на train и validation, а сюда приходит зафиксированным.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np

from src import contracts as C
from src.assets import (
    PLACEMENT_RULE,
    PROBABILITY_RULE,
    evaluate_assets,
    place_assets,
    proxy_error,
    total_expected_loss,
    unassessed_exposure,
)
from src.data import chips as chips_mod
from src.data import fetch
from src.export.bundle import assemble, new_run_id
from src.methods.baseline_threshold import BaselineThreshold
from src.methods.main_model import MainModel
from src.procurement.candidates import CatalogConfig, build_catalog, selftest_catalog
from src.procurement.pricing import position_row
from src.procurement.strategies import StrategyConfig, build_strategies
from src.sensitivity import run_sensitivity
from src.sensitivity_ranks import COLUMNS as RANK_COLUMNS, run as run_rank_sensitivity


def priority_raster(prob: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    """Где дополнительная съёмка полезнее всего.

    Высокий приоритет там, где вероятность затопления заметна И оценка при этом
    ненадёжна: снимок купленной зоны должен приносить информацию, а не подтверждать
    очевидное. Сухая уверенная суша и уверенно залитая вода приоритета не получают.
    """
    return (np.clip(prob, 0, 1) * np.clip(uncertainty, 0, 1)).astype(np.float32)


def event_date(event: str) -> date | None:
    """Дата съёмки события из Sen1Floods11_Metadata.geojson официального набора."""
    path = fetch.DEFAULT_ROOT / "Sen1Floods11_Metadata.geojson"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        if str(props.get("location", "")).lower() == event.lower():
            raw = str(props.get("s1_date", ""))
            try:
                return date(*(int(part) for part in raw.split("/")))
            except (TypeError, ValueError):
                return None
    return None


def scenario_dates(event: str, deadline: str) -> dict[str, str]:
    """Сценарные даты заказа, привязанные к дате самого события.

    Оперативный горизонт кейса — 6–12 часов, поэтому решение принимается на следующие
    сутки после съёмки события, а новая съёмка планируется к этому же сроку. Архивный
    контекст берётся старше 90 дней, иначе коэффициент актуальности был бы другим.
    Все даты сценарные, и это записано в паспорте запуска.
    """
    base = event_date(event) or date.today()
    decision = deadline or (base + timedelta(days=1)).isoformat()
    return {
        "event_date": base.isoformat(),
        "decision_deadline": decision,
        "new_observation_at": (base + timedelta(days=1)).isoformat(),
        "new_available_at": decision,
        "archive_observation_at": (base - timedelta(days=180)).isoformat(),
        "archive_available_at": decision,
    }


def build_source_manifest(chip_id: str, chip_path: Path) -> list[dict]:
    event = chips_mod.event_of(chip_id)
    observed = event_date(event)
    observed_at = observed.isoformat() if observed else ""
    return [
        {
            "id": f"{chip_id}_S1Hand",
            "file": chip_path.as_posix(),
            "url": f"{C.DATASET_BUCKET_HTTPS}/v1.1/data/flood_events/HandLabeled/S1Hand/{chip_id}_S1Hand.tif",
            "dataset": f"{C.DATASET_NAME} {C.DATASET_VERSION}",
            "event": event,
            "observation_date": observed_at,
            "published": "2020",
            "license": "Cloud to Street, открытый доступ, цитирование обязательно",
            "purpose": "обязательный исходный снимок, единственный вход основного метода",
            "limits": "два канала VV/VH в дБ, 512×512, шаг 10 м",
        },
        {
            "id": f"{chip_id}_LabelHand",
            "file": fetch.layer_path(chip_id, C.LAYER_LABEL).as_posix(),
            "observation_date": observed_at,
            "url": f"{C.DATASET_BUCKET_HTTPS}/v1.1/data/flood_events/HandLabeled/LabelHand/{chip_id}_LabelHand.tif",
            "dataset": f"{C.DATASET_NAME} {C.DATASET_VERSION}",
            "published": "2020",
            "license": "Cloud to Street",
            "purpose": "ручная разметка воды для обучения и независимой проверки",
            "limits": "−1 нет метки, 0 не вода, 1 вода; временную и постоянную воду не различает",
        },
        {
            "id": f"{chip_id}_JRCWaterHand",
            "file": fetch.layer_path(chip_id, C.LAYER_JRC).as_posix(),
            "observation_date": "1984-2020 (период наблюдения JRC Global Surface Water)",
            "url": f"{C.DATASET_BUCKET_HTTPS}/v1.1/data/flood_events/HandLabeled/JRCWaterHand/{chip_id}_JRCWaterHand.tif",
            "dataset": f"{C.DATASET_NAME} {C.DATASET_VERSION}",
            "source": "JRC Global Surface Water, Landsat 1984–2020",
            "published": "2020",
            "license": "Cloud to Street / EC JRC, открытый доступ",
            "purpose": (
                "постоянная вода: используется на стороне разметки для выделения целевого "
                "класса «временное затопление». На применении не требуется"
            ),
            "limits": (
                "производный продукт по оптике, ошибается на узких реках и мелководье; "
                "построен до событий набора, то есть временно корректен"
            ),
        },
        {
            "id": "citation",
            "url": C.DATASET_REPO,
            "purpose": "официальный источник набора",
            "license": C.DATASET_CITATION,
        },
    ]


def _split_part_of(chip_id: str, split_dir: Path = Path("splits")) -> str:
    """К какой части сплита относится чип. Жюри сверит это с составом обучения."""
    for part in ("train", "validation", "test", "holdout"):
        path = split_dir / f"{part}.csv"
        if path.exists() and chip_id in path.read_text(encoding="utf-8"):
            return part
    return "вне сплита"


def independent_proxy_check(
    model, seed: int, chips_count: int = 5, split_dir: Path = Path("splits")
) -> dict:
    """Прокси-проверка ущерба на чипах, которых модель не видела.

    Демонстрационный чип выбирается по наглядности и вполне может лежать в обучающей
    части. Проверять на нём качество перехода «вероятность → рубли» — значит мерить
    себя по своей же выборке. Постановка это предусматривает прямо: если у
    демонстрационного чипа нет пригодной независимой разметки, проверку выполняют на
    отдельном размеченном тестовом чипе, не использованном для настройки модели и
    порогов.

    Правило выбора контрольных точек: берутся первые ``chips_count`` чипов части test
    в порядке сплита, на каждом размещается портфель из десяти объектов тем же
    правилом и тем же зерном, что и демонстрационный, с добавлением номера чипа.
    Точки, попавшие в пиксели без надёжной ручной метки, из проверки исключаются и
    считаются отдельно — метка −1 это не «сухо», а отсутствие метки. Подбора точек по
    прогнозу нет нигде: портфели размещаются до расчёта.
    """
    import numpy as _np

    manifest = split_dir / "split_manifest.json"
    if not manifest.exists():
        return {"status": "not_checked", "note": "нет сплита, независимые чипы не выбрать"}

    from src.eval.runner import part_chip_ids

    try:
        candidates = part_chip_ids("test", split_dir)
    except (KeyError, FileNotFoundError):
        return {"status": "not_checked", "note": "в сплите нет части test"}
    if not candidates:
        return {"status": "not_checked", "note": "часть test пуста"}

    predicted: list[float] = []
    reference: list[float] = []
    used_chips: list[str] = []
    skipped = 0

    for index, chip_id in enumerate(candidates[:chips_count]):
        try:
            chip = chips_mod.load_chip(chip_id)
        except chips_mod.ChipError:
            continue
        if chip.jrc is None or chip.label is None:
            continue

        prob, uncertainty = model.predict_chip(chip.vv, chip.vh)
        valid_s1 = _np.isfinite(chip.vv) & _np.isfinite(chip.vh)
        prob_out = prob.astype(_np.float32).copy()
        prob_out[~valid_s1] = C.PROB_NODATA

        assets = place_assets(chip_id, chip.vv, chip.vh, chip.transform, seed + index)
        rows = evaluate_assets(assets, prob_out, uncertainty)
        report = proxy_error(
            rows, assets, chips_mod.target_flood(chip), label_valid=chips_mod.valid_mask(chip)
        )
        skipped += int(report.get("n_skipped_without_label", 0))
        if report.get("status") != "checked":
            continue

        used_chips.append(chip_id)
        by_id = {a.asset_id: a for a in assets}
        for row in rows:
            if row["status"] != C.ASSET_STATUS_OK:
                continue
            asset = by_id[row["asset_id"]]
            if not chips_mod.valid_mask(chip)[asset.row, asset.col]:
                continue
            y = float(bool(chips_mod.target_flood(chip)[asset.row, asset.col]))
            predicted.append(float(row["expected_loss_rub"]))
            reference.append(C.expected_loss(y, asset.asset_value_rub, asset.vulnerability_coef))

    if not predicted:
        return {
            "status": "not_checked",
            "note": "на тестовых чипах не нашлось контрольных точек с надёжной меткой",
            "n_skipped_without_label": skipped,
        }

    p_arr = _np.array(predicted)
    r_arr = _np.array(reference)
    return {
        "status": "checked",
        "split_part": "test",
        "chips": used_chips,
        "seed": seed,
        "n_points": int(p_arr.size),
        "n_skipped_without_label": skipped,
        "mae_rub": float(_np.mean(_np.abs(p_arr - r_arr))),
        "rmse_rub": float(_np.sqrt(_np.mean((p_arr - r_arr) ** 2))),
        "bias_rub": float(_np.mean(p_arr - r_arr)),
        "sum_predicted_rub": float(p_arr.sum()),
        "sum_reference_rub": float(r_arr.sum()),
        "rule": (
            "Первые чипы части test, по десять объектов на чип, размещение тем же "
            "правилом и зерном, что у демонстрационного портфеля. Точки без надёжной "
            "ручной метки исключены и посчитаны отдельно."
        ),
        "why": (
            "Эти чипы не участвовали ни в обучении, ни в калибровке, ни в выборе порога. "
            "Именно это число и следует считать оценкой качества перехода "
            "«вероятность → рубли»."
        ),
    }


def _budget(raw: str) -> float:
    """Бюджет проверяется до расчёта, а не после: иначе пайплайн отработает впустую."""
    try:
        value = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"бюджет должен быть числом, получено {raw!r}") from error
    if value < 0:
        raise argparse.ArgumentTypeError("бюджет не может быть отрицательным")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", required=True, help="идентификатор чипа, например India_900498")
    parser.add_argument("--budget", type=_budget, required=True, help="объявленный бюджет, руб.")
    parser.add_argument("--model", type=Path, default=Path("models/main"))
    parser.add_argument("--baseline", type=Path, default=Path("models/baseline.json"))
    parser.add_argument("--seed", type=int, default=2026, help="зерно размещения объектов")
    parser.add_argument("--other-cost", type=float, default=0.0)
    parser.add_argument("--deadline", default="", help="срок принятия решения, ГГГГ-ММ-ДД")
    parser.add_argument("--cell-km", type=float, default=1.2)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    print(f"чип: {args.chip}, бюджет: {args.budget:,.2f} руб.", flush=True)
    try:
        chip = chips_mod.load_chip(args.chip)
    except chips_mod.ChipError as error:
        # Понятная одна строка вместо трейсбека: чип задаёт человек, ошибается тоже человек.
        raise SystemExit(str(error))
    chip_path = fetch.layer_path(args.chip, C.LAYER_S1)

    # ── вероятность и маска ──────────────────────────────────────────────────
    model = MainModel.load(args.model)
    if model.threshold is None:
        raise SystemExit("у модели не зафиксирован порог: сначала python -m src.cli.train_main")
    prob, uncertainty = model.predict_chip(chip.vv, chip.vh)
    s1_valid = np.isfinite(chip.vv) & np.isfinite(chip.vh)

    prob_out = prob.astype(np.float32).copy()
    prob_out[~s1_valid] = C.PROB_NODATA
    mask_out = np.full(prob.shape, C.MASK_NODATA, dtype=np.uint8)
    mask_out[s1_valid] = np.where(prob[s1_valid] >= model.threshold, C.MASK_WATER, C.MASK_DRY)
    print(
        f"  вероятность посчитана, порог {model.threshold:.2f}, "
        f"вода на {int((mask_out == C.MASK_WATER).sum()):,} пикселях",
        flush=True,
    )

    # ── десять объектов и ущерб ──────────────────────────────────────────────
    assets = place_assets(args.chip, chip.vv, chip.vh, chip.transform, args.seed)
    asset_rows = evaluate_assets(assets, prob_out, uncertainty)
    total_el = total_expected_loss(asset_rows)
    unassessed = unassessed_exposure(asset_rows, assets)
    print(f"  объектов оценено: {C.ASSET_COUNT - unassessed['count']}, ущерб {total_el:,.0f} руб.", flush=True)

    independent_proxy = independent_proxy_check(model, args.seed)
    if independent_proxy.get("status") == "checked":
        print(
            f"  прокси-проверка на {len(independent_proxy['chips'])} независимых чипах: "
            f"MAE {independent_proxy['mae_rub']:,.0f} руб. по "
            f"{independent_proxy['n_points']} точкам "
            f"(исключено без метки: {independent_proxy['n_skipped_without_label']})",
            flush=True,
        )
    else:
        print(f"  независимая прокси-проверка: {independent_proxy.get('note', '')}", flush=True)

    proxy = (
        proxy_error(
            asset_rows,
            assets,
            chips_mod.target_flood(chip),
            # Пиксели без надёжной ручной метки из проверки исключаются: метка −1 это
            # не «сухо», а отсутствие метки.
            label_valid=chips_mod.valid_mask(chip),
        )
        if chip.jrc is not None
        else {"status": "not_checked", "note": "нет слоя постоянной воды, проверка невозможна"}
    )

    # ── зоны и стратегии ─────────────────────────────────────────────────────
    dates = scenario_dates(chips_mod.event_of(args.chip), args.deadline)
    catalog = build_catalog(
        priority_raster(prob, uncertainty),
        chip.transform,
        {a.asset_id: (a.longitude, a.latitude) for a in assets},
        CatalogConfig(
            cell_km=args.cell_km,
            observation_at=dates["new_observation_at"],
            available_at=dates["new_available_at"],
            context_observation_at=dates["archive_observation_at"],
            context_available_at=dates["archive_available_at"],
        ),
    )
    problems = selftest_catalog(catalog)
    if problems:
        raise SystemExit("каталог зон не прошёл самопроверку: " + "; ".join(problems))
    print(f"  кандидатных зон: {len(catalog)}", flush=True)

    assets_geojson = {"type": "FeatureCollection", "features": [a.as_feature() for a in assets]}
    config = StrategyConfig(
        budget_rub=Decimal(str(args.budget)),
        other_cost_rub=Decimal(str(args.other_cost)),
        decision_deadline=dates["decision_deadline"],
    )
    strategies = build_strategies(catalog, assets_geojson, asset_rows, config)
    for row in strategies["comparison"]:
        print(
            f"    {row['strategy']}: {len(strategies['plans'][row['strategy']])} зон, "
            f"{float(row['data_cost_rub']):,.2f} руб., покрыто "
            f"{row['covered_expected_loss_rub']:,.0f} руб."
            + ("" if row["budget_feasible"] else "  [контрфактическая]"),
            flush=True,
        )

    sensitivity = run_sensitivity(catalog, assets_geojson, asset_rows, config)
    rank_rows, rank_notes = run_rank_sensitivity(
        asset_rows, {a.asset_id: a.asset_class for a in assets}
    )
    print(f"  сценариев чувствительности: {len(sensitivity)}", flush=True)
    print(f"  сценариев сдвига приоритетов: {len(set(r['scenario_id'] for r in rank_rows))}", flush=True)
    for note in rank_notes:
        print(f"    {note}", flush=True)

    # ── паспорт запуска ──────────────────────────────────────────────────────
    run_id = new_run_id(args.chip, args.budget)
    baseline = BaselineThreshold.load(args.baseline) if args.baseline.exists() else None
    metadata = {
        "run_id": run_id,
        "chip_id": args.chip,
        "event_id": chips_mod.event_of(args.chip),
        # posix-разделители: в контейнере на Linux путь с обратными слешами
        # превращается в имя одного файла и подложка карты не открывается.
        "source_chip_path": chip_path.as_posix(),
        "s1_path": chip_path.as_posix(),
        "threshold": model.threshold,
        "budget_rub": args.budget,
        "decision_deadline": dates["decision_deadline"],
        "scenario_dates": dates,
        "dates_status": (
            "сценарные: привязаны к дате съёмки события из Sen1Floods11_Metadata.geojson, подтверждением поставщика не являются"
        ),
        "target_class": model.config.target_name,
        "permanent_water_source": "JRCWaterHand из официального набора, только на стороне разметки",
        "inference_inputs": "только Sentinel-1 VV/VH, дополнительные слои не требуются",
        "seed": args.seed,
        "placement_rule": PLACEMENT_RULE,
        "probability_rule": PROBABILITY_RULE,
        "loss_model": "EL = p_flood × V × q, p из flood_probability.tif независимо от маски",
        "chip_split_part": _split_part_of(args.chip),
        "split": json.loads(Path("splits/split_manifest.json").read_text(encoding="utf-8"))
        if Path("splits/split_manifest.json").exists()
        else {},
        "methods": {
            "baseline": (
                {
                    "kind": "пороговый по Sentinel-1",
                    "channel": baseline.config.channel,
                    "threshold_db": baseline.config.threshold_db,
                    "despeckle_size": baseline.config.despeckle_size,
                    "selection": baseline.config.selection,
                }
                if baseline
                else None
            ),
            "main": {
                "kind": "ансамбль LightGBM по признакам S1 с пространственным контекстом",
                "features": model.feature_names,
                "ensemble_size": len(model.boosters),
                "calibration": "изотоническая, на validation",
                "trained_on_chips": len(model.trained_on),
                "trained_on_note": (
                    "Число обученных чипов может быть меньше числа чипов в train: чипы, где "
                    "после отбора валидных пикселей не осталось ни одного пригодного образца, "
                    "в обучение не попадают. Это не потеря данных, а отсев пустых кадров."
                ),
                "threshold_selected_on": "validation",
            },
        },
        "uncertainty": {
            "definition": "0,7 × нормированная энтропия вероятности + 0,3 × разброс ансамбля",
            "units": "безразмерная величина от 0 до 1",
            "checked": "AUC связи с ошибками, см. reports/metrics_main_validation.json",
        },
        "pricing": {
            "formula": C.PRICE_FORMULA,
            "legal_edition": C.LEGAL_EDITION,
            "legal_check_date": C.LEGAL_CHECK_DATE,
            "base_rate_rub_km2": float(C.BASE_RATE_RUB_KM2_SCENARIO),
            "base_rate_status": C.BASE_RATE_STATUS_SCENARIO,
            "base_rate_source": C.BASE_RATE_SOURCE_SCENARIO,
            "rounding": "площадь 0,001 км², скидка 9 знаков, деньги до копеек, ROUND_HALF_UP",
            "discount_grouping": "тип данных + разрешение + календарный год",
        },
        "zones": {
            "rule": f"сетка ячеек {args.cell_km} км по приоритету «вероятность × неопределённость»",
            "min_area_km2": C.MIN_ORDER_AREA_KM2,
            "availability": "даты сценарные, привязаны к дате события; см. scenario_dates",
            "broad_rule_B": config.broad_rule,
        },
        "total_expected_loss_rub": total_el,
        "unassessed_exposure": unassessed,
        "proxy_check": proxy,
        "proxy_check_note": (
            "Демонстрационный чип может входить в обучающую часть — он выбирается по "
            "наглядности. Поэтому рядом лежит проверка на независимом чипе из части "
            "test; именно её и следует считать оценкой качества перехода "
            "«вероятность → рубли»."
        ),
        "proxy_check_independent": independent_proxy,
        "rank_sensitivity": {"file": "sensitivity_ranks.csv", "notes": rank_notes},
        "bounds": _bounds_of(chip.transform, prob.shape),
        "commit": _git_commit(),
        "generated_at": date.today().isoformat(),
        "reproduce": [
            "python -m src.cli.fetch_data",
            "python -m src.cli.make_splits",
            "python -m src.cli.train_baseline",
            "python -m src.cli.train_main",
            f"python -m src.cli.run_bundle --chip {args.chip} --budget {args.budget:g}",
        ],
    }

    profile = dict(chip.profile)
    run_dir = args.outputs / run_id
    paths = assemble(
        run_dir,
        {
            "profile": profile,
            "probability": prob_out,
            "mask": mask_out,
            "assets": [a.as_row() for a in assets],
            "asset_loss": asset_rows,
            "candidates": catalog,
            "procurement": [position_row(p) for p in strategies["positions"]],
            "strategy_plans": strategies["plans"],
            "strategy_comparison": strategies["comparison"],
            "sensitivity": sensitivity,
            "run_metadata": metadata,
            "source_manifest": build_source_manifest(args.chip, chip_path),
        },
    )

    # Сдвиг приоритетов при изменении p, V и q — критерий 14. В обязательный состав
    # пакета файл не входит, но без него влияние на приоритеты показать нечем.
    _write_rank_sensitivity(run_dir / "sensitivity_ranks.csv", rank_rows)

    # Карта неопределённости не входит в обязательный пакет, но нужна интерфейсу.
    _write_uncertainty(run_dir / "uncertainty.tif", uncertainty, s1_valid, profile)
    print(f"\nпакет собран: {paths.run_dir}", flush=True)

    if not args.skip_validate:
        print("\nпроверка пакета:", flush=True)
        code = subprocess.call([sys.executable, "-m", "src.validate_bundle", str(run_dir)])
        if code != 0:
            raise SystemExit(code)


def _write_rank_sensitivity(path: Path, rows: list[dict]) -> None:
    """CSV сдвига приоритетов. Пустое значение остаётся пустым, нулём не заменяется."""
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RANK_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in RANK_COLUMNS})


def _bounds_of(transform, shape: tuple[int, int]) -> list[float]:
    """Границы растра [запад, юг, восток, север] — их ждёт карта в интерфейсе."""
    height, width = shape
    west, north = transform * (0, 0)
    east, south = transform * (width, height)
    return [float(west), float(south), float(east), float(north)]


def _write_uncertainty(path: Path, uncertainty: np.ndarray, valid: np.ndarray, profile: dict) -> None:
    import rasterio

    data = uncertainty.astype(np.float32).copy()
    data[~valid] = C.PROB_NODATA
    out = dict(profile)
    out.update(driver="GTiff", dtype="float32", count=1, nodata=C.PROB_NODATA, compress="deflate")
    out.pop("bounds", None)
    with rasterio.open(path, "w", **out) as dataset:
        dataset.write(data, 1)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
