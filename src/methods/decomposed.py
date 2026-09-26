"""Основной метод во втором исполнении: задача разложена на две головы.

Зачем понадобилось. Сравнение на независимой проверке показало неприятное: наша
модель, обученная напрямую на редком классе «временное затопление», потеряла
способность находить воду вообще. На тесте пороговый baseline по цели «вода вообще»
даёт F1 0,671, а модель — 0,462. То есть, пока модель училась вычитать постоянную
воду, она разучилась делать более простую и более важную вещь.

Разложение возвращает обе способности и учит каждую отдельно:

    p(затопление) = p(вода) × (1 − p(постоянная | вода))

Первая голова решает простую задачу с частым классом: где вообще вода. Класс
встречается в разы чаще, сигнал физический — гладкая вода темна на радаре, — и такая
задача заметно лучше переносится на новые события.

Вторая голова решает задачу, для которой у нас есть прямая разметка из официального
набора: какая из этой воды постоянная. Здесь целевой класс тоже частый, а признаки
другие — постоянные водоёмы крупнее, компактнее и имеют устойчивую форму.

Перемножение даёт вероятность временного затопления. Калибруется произведение целиком,
на validation, изотонической регрессией — так же, как в одноголовом варианте, чтобы
сравнение было честным.

Обе головы учатся только по Sentinel-1. Слой постоянной воды нужен исключительно как
разметка второй головы при обучении; на применении модели не требуется ничего, кроме
одного радарного снимка.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from src.methods.features import (
    FeatureConfig,
    build_features,
    feature_names,
    sample_pixels,
    sample_pixels_uniform,
)
from src.methods.main_model import stable_hash

#: Один образец: чип, каналы, вода, постоянная вода, валидность.
Sample = tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass
class DecomposedConfig:
    """Гиперпараметры обеих голов. Подбираются на validation."""

    n_estimators: int = 400
    learning_rate: float = 0.06
    num_leaves: int = 63
    min_child_samples: int = 80
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    ensemble_size: int = 3
    pixels_per_chip: int = 4000
    calibration_pixels: int = 20_000
    #: Доля положительных для головы воды. Вода встречается часто, сильный перекос не нужен.
    water_positive_share: float = 0.4
    #: Голова постоянной воды учится только по пикселям воды, там классы сбалансированнее.
    permanent_positive_share: float = 0.5
    event_balanced: bool = True
    seed: int = 42
    features: FeatureConfig = field(default_factory=FeatureConfig)
    target_name: str = "временное затопление (разложение на две головы)"

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["features"] = {
            "scales": list(self.features.scales),
            "use_chip_contrast": self.features.use_chip_contrast,
            "use_chip_absolute": self.features.use_chip_absolute,
            "despeckle_sizes": list(self.features.despeckle_sizes),
            "per_chip_normalize": self.features.per_chip_normalize,
        }
        return payload


class DecomposedModel:
    """Две головы LightGBM плюс общая изотоническая калибровка произведения."""

    def __init__(self, config: DecomposedConfig | None = None) -> None:
        self.config = config or DecomposedConfig()
        self.water_boosters: list = []
        self.permanent_boosters: list = []
        self.calibrator = None
        self.feature_names: list[str] = feature_names(self.config.features)
        self.threshold: float | None = None
        self.trained_on: list[str] = []
        self.calibrated_on: list[str] = []

    # ── обучение ─────────────────────────────────────────────────────────────

    def _make_booster(self, seed: int):
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            subsample=self.config.subsample,
            subsample_freq=1,
            colsample_bytree=self.config.colsample_bytree,
            reg_lambda=self.config.reg_lambda,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    def _balance_by_event(self, chips, rows, labels):
        """Выравнивает пиксельный вклад событий: крупные события иначе перетягивают выборку."""
        from collections import defaultdict

        by_event: dict[str, list[int]] = defaultdict(list)
        for index, chip_id in enumerate(chips):
            by_event[chip_id.split("_")[0]].append(index)
        if len(by_event) < 2:
            return rows, labels
        budget = min(sum(len(rows[i]) for i in idx) for idx in by_event.values())
        rng = np.random.default_rng(self.config.seed + 991)
        new_rows, new_labels = [], []
        for indexes in by_event.values():
            per_chip = max(1, budget // len(indexes))
            for index in indexes:
                take = min(per_chip, len(rows[index]))
                picked = rng.choice(len(rows[index]), size=take, replace=False)
                new_rows.append(rows[index][picked])
                new_labels.append(labels[index][picked])
        return new_rows, new_labels

    def fit(self, samples: Iterable[Sample]) -> "DecomposedModel":
        """samples: (chip_id, vv, vh, water, permanent, valid) — только обучающая часть."""
        water_rows: list[np.ndarray] = []
        water_labels: list[np.ndarray] = []
        perm_rows: list[np.ndarray] = []
        perm_labels: list[np.ndarray] = []
        chips: list[str] = []

        for chip_id, vv, vh, water, permanent, valid in samples:
            cube = build_features(vv, vh, self.config.features)
            seed = self.config.seed + stable_hash(chip_id) % 10_000
            rng = np.random.default_rng(seed)

            x, y = sample_pixels(
                cube, water, valid, self.config.pixels_per_chip, rng,
                self.config.water_positive_share,
            )
            if not x.size:
                continue
            water_rows.append(x)
            water_labels.append(y)
            chips.append(chip_id)

            # Вторая голова учится ТОЛЬКО по пикселям воды: вопрос «постоянная ли эта
            # вода» на суше не имеет смысла и только размывал бы задачу.
            water_only = valid & water
            if water_only.any():
                xp, yp = sample_pixels(
                    cube, permanent, water_only, self.config.pixels_per_chip,
                    np.random.default_rng(seed + 1), self.config.permanent_positive_share,
                )
                if xp.size:
                    perm_rows.append(xp)
                    perm_labels.append(yp)

        if not water_rows:
            raise ValueError("обучающая выборка пуста")
        if not perm_rows:
            raise ValueError("нет пикселей воды для головы постоянной воды")

        if self.config.event_balanced:
            water_rows, water_labels = self._balance_by_event(chips, water_rows, water_labels)

        self.trained_on = chips
        self.water_boosters = self._fit_heads(water_rows, water_labels, offset=0)
        self.permanent_boosters = self._fit_heads(perm_rows, perm_labels, offset=100)
        return self

    def _fit_heads(self, rows, labels, offset: int) -> list:
        features = np.concatenate(rows).astype(np.float32)
        answers = np.concatenate(labels).astype(np.int8)
        boosters = []
        for index in range(self.config.ensemble_size):
            seed = self.config.seed + offset + index
            rng = np.random.default_rng(seed)
            take = rng.choice(len(features), size=int(0.8 * len(features)), replace=False)
            booster = self._make_booster(seed)
            booster.fit(features[take], answers[take])
            boosters.append(booster)
        return boosters

    def calibrate(self, samples: Iterable[Sample]) -> dict[str, float]:
        """Калибруется произведение голов, на validation, с природными долями классов."""
        from sklearn.isotonic import IsotonicRegression

        raw: list[np.ndarray] = []
        answers: list[np.ndarray] = []
        chips: list[str] = []
        rng = np.random.default_rng(self.config.seed + 777)

        for chip_id, vv, vh, water, permanent, valid in samples:
            cube = build_features(vv, vh, self.config.features)
            flood = water & ~permanent
            x, y = sample_pixels_uniform(cube, flood, valid, self.config.calibration_pixels, rng)
            if not x.size:
                continue
            raw.append(self._raw_product(x))
            answers.append(y)
            chips.append(chip_id)

        if not raw:
            raise ValueError("валидационная выборка пуста, калибровать нечего")

        p = np.concatenate(raw)
        y = np.concatenate(answers)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(p, y)
        self.calibrated_on = chips

        from src.eval.metrics import brier_score, expected_calibration_error

        return {
            "brier_raw": brier_score(p, y),
            "brier_calibrated": brier_score(self.calibrator.predict(p), y),
            "ece_raw": expected_calibration_error(p, y),
            "ece_calibrated": expected_calibration_error(self.calibrator.predict(p), y),
            "n_pixels": int(p.size),
        }

    # ── применение ───────────────────────────────────────────────────────────

    def _stack(self, boosters: list, features: np.ndarray) -> np.ndarray:
        return np.stack([b.predict_proba(features)[:, 1] for b in boosters])

    def _raw_product(self, features: np.ndarray) -> np.ndarray:
        water = self._stack(self.water_boosters, features).mean(axis=0)
        permanent = self._stack(self.permanent_boosters, features).mean(axis=0)
        return water * (1.0 - permanent)

    def predict_parts(self, vv: np.ndarray, vh: np.ndarray) -> dict[str, np.ndarray]:
        """Обе головы по отдельности — для отчёта и для разбора ошибок на защите."""
        cube = build_features(vv, vh, self.config.features)
        height, width, n_features = cube.shape
        flat = cube.reshape(-1, n_features)
        water_stack = self._stack(self.water_boosters, flat)
        perm_stack = self._stack(self.permanent_boosters, flat)
        return {
            "water": water_stack.mean(axis=0).reshape(height, width),
            "permanent": perm_stack.mean(axis=0).reshape(height, width),
            "water_spread": water_stack.std(axis=0).reshape(height, width),
            "permanent_spread": perm_stack.std(axis=0).reshape(height, width),
        }

    def predict_chip(self, vv: np.ndarray, vh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Вероятность временного затопления и неопределённость, обе (H, W) float32.

        Работает по одному Sentinel-1: слой постоянной воды нужен был только при
        обучении второй головы.
        """
        if not self.water_boosters or not self.permanent_boosters:
            raise RuntimeError("модель не обучена")
        parts = self.predict_parts(vv, vh)
        raw = parts["water"] * (1.0 - parts["permanent"])
        prob = self.calibrator.predict(raw.ravel()) if self.calibrator is not None else raw.ravel()
        prob = np.clip(prob, 0.0, 1.0).astype(np.float32).reshape(raw.shape)

        eps = 1e-6
        p = np.clip(prob, eps, 1 - eps)
        entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        # Разброс обеих голов складывается: неуверенность в любой из них делает
        # произведение ненадёжным.
        spread = parts["water_spread"] + parts["permanent_spread"]
        uncertainty = np.clip(0.7 * entropy + 0.3 * (spread / 0.5), 0.0, 1.0).astype(np.float32)
        return prob, uncertainty

    def predict_mask(self, prob: np.ndarray, threshold: float | None = None) -> np.ndarray:
        value = self.threshold if threshold is None else threshold
        if value is None:
            raise RuntimeError("порог не выбран: сначала выберите его на validation")
        return np.asarray(prob) >= float(value)

    # ── сохранение ───────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "main_model.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "water": self.water_boosters,
                    "permanent": self.permanent_boosters,
                    "calibrator": self.calibrator,
                },
                handle,
                protocol=5,
            )
        meta = {
            "kind": "decomposed",
            "config": self.config.to_json(),
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "trained_on": self.trained_on,
            "calibrated_on": self.calibrated_on,
        }
        (directory / "main_model.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "DecomposedModel":
        directory = Path(directory)
        meta = json.loads((directory / "main_model.json").read_text(encoding="utf-8"))
        payload = dict(meta["config"])
        features_payload = payload.pop("features", {})
        config = DecomposedConfig(
            **payload,
            features=FeatureConfig(
                scales=tuple(features_payload.get("scales", ())),
                use_chip_contrast=features_payload.get("use_chip_contrast", True),
                use_chip_absolute=features_payload.get("use_chip_absolute", False),
                despeckle_sizes=tuple(features_payload.get("despeckle_sizes", ())),
                per_chip_normalize=features_payload.get("per_chip_normalize", False),
            ),
        )
        model = cls(config)
        with (directory / "main_model.pkl").open("rb") as handle:
            blob = pickle.load(handle)
        model.water_boosters = blob["water"]
        model.permanent_boosters = blob["permanent"]
        model.calibrator = blob["calibrator"]
        model.feature_names = meta["feature_names"]
        model.threshold = meta.get("threshold")
        model.trained_on = meta.get("trained_on", [])
        model.calibrated_on = meta.get("calibrated_on", [])
        return model

    def feature_importance(self) -> dict[str, list[tuple[str, float]]]:
        """Важность признаков по каждой голове отдельно — они решают разные задачи."""
        out: dict[str, list[tuple[str, float]]] = {}
        for name, boosters in (("water", self.water_boosters), ("permanent", self.permanent_boosters)):
            if not boosters:
                out[name] = []
                continue
            total = np.mean([b.feature_importances_ for b in boosters], axis=0)
            share = total / total.sum() if total.sum() else total
            out[name] = sorted(zip(self.feature_names, share), key=lambda item: -item[1])[:10]
        return out
