"""Выгрузка нужных слоёв Sen1Floods11 из официального бакета.

Бакет публичный, поэтому работаем обычным HTTPS без gsutil и без учётных данных.
Качаем только размеченную часть — 446 чипов в четырёх слоях, это около 0,5 ГБ,
а не 14 ГБ полного набора со слабой разметкой.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Iterable

from src.contracts import (
    DATASET_BUCKET_HTTPS,
    DATASET_VERSION,
    HAND_LAYERS,
)

HAND_PREFIX = f"{DATASET_VERSION}/data/flood_events/HandLabeled"
SPLIT_PREFIX = f"{DATASET_VERSION}/splits/flood_handlabeled"
LIST_API = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"

AUTHOR_SPLITS = (
    "flood_train_data.csv",
    "flood_valid_data.csv",
    "flood_test_data.csv",
    "flood_bolivia_data.csv",
)

DEFAULT_ROOT = Path("data/cache")


def _get(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def list_layer(layer: str) -> list[str]:
    """Имена файлов слоя в бакете, по одной странице на 1000 объектов."""
    names: list[str] = []
    token = ""
    while True:
        url = f"{LIST_API}?prefix={HAND_PREFIX}/{layer}/&maxResults=1000&fields=items(name),nextPageToken"
        if token:
            url += f"&pageToken={token}"
        page = json.loads(_get(url))
        names.extend(item["name"].split("/")[-1] for item in page.get("items", []))
        token = page.get("nextPageToken", "")
        if not token:
            break
    return sorted(n for n in names if n.endswith(".tif"))


def chip_ids(layer: str = "S1Hand") -> list[str]:
    """Идентификаторы чипов вида ``India_900498``."""
    suffix = f"_{layer}.tif"
    return [n[: -len(suffix)] for n in list_layer(layer)]


def layer_url(chip_id: str, layer: str) -> str:
    return f"{DATASET_BUCKET_HTTPS}/{HAND_PREFIX}/{layer}/{chip_id}_{layer}.tif"


def layer_path(chip_id: str, layer: str, root: Path = DEFAULT_ROOT) -> Path:
    return root / layer / f"{chip_id}_{layer}.tif"


def fetch_one(chip_id: str, layer: str, root: Path = DEFAULT_ROOT) -> Path:
    """Скачивает слой чипа, если его ещё нет. Возвращает путь к файлу."""
    dest = layer_path(chip_id, layer, root)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = _get(layer_url(chip_id, layer))
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(payload)
    tmp.replace(dest)
    return dest


def fetch_layers(
    chips: Iterable[str],
    layers: Iterable[str] = HAND_LAYERS,
    root: Path = DEFAULT_ROOT,
    workers: int = 12,
) -> dict[str, int]:
    """Параллельная выгрузка. Возвращает число файлов по слоям."""
    jobs = [(chip, layer) for chip in chips for layer in layers]
    done: dict[str, int] = {layer: 0 for layer in layers}
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, chip, layer, root): layer for chip, layer in jobs}
        for future in cf.as_completed(futures):
            future.result()
            done[futures[future]] += 1
    return done


def fetch_author_splits(root: Path = DEFAULT_ROOT) -> Path:
    """Официальные split-файлы авторов. Нужны для сравнения с нашим протоколом."""
    out = root / "author_splits"
    out.mkdir(parents=True, exist_ok=True)
    for name in AUTHOR_SPLITS:
        dest = out / name
        if not dest.exists():
            dest.write_bytes(_get(f"{DATASET_BUCKET_HTTPS}/{SPLIT_PREFIX}/{name}"))
    return out


def fetch_metadata(root: Path = DEFAULT_ROOT) -> Path:
    """Sen1Floods11_Metadata.geojson — даты съёмки и орбиты по событиям."""
    dest = root / "Sen1Floods11_Metadata.geojson"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_get(f"{DATASET_BUCKET_HTTPS}/{DATASET_VERSION}/Sen1Floods11_Metadata.geojson"))
    return dest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
