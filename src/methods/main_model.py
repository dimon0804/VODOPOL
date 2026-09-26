"""Основной метод: бустинг по признакам Sentinel-1 с пространственным контекстом.

Чем он отличается от baseline, а не просто «другой порог»: обучаемая модель в
многомерном признаковом пространстве, калиброванные вероятности и проверяемая
оценка неопределённости. Порог применяется ПОСЛЕ модели и только для того, чтобы
получить согласованную бинарную маску.

Вероятности калибруются изотонической регрессией на validation — так значение 0,7
начинает означать наблюдаемую долю воды около 0,7, а не просто «модель уверена».

Неопределённость складывается из двух источников:
  * энтропия калиброванной вероятности — насколько предсказание близко к границе;
  * разброс ансамбля из нескольких моделей, обученных на разных подвыборках, —
    насколько результат зависит от того, какие пиксели попали в обучение.
Полезность этого показателя проверяется отдельно (`src.eval.metrics`), и если
проверка не подтверждается, в файлы идёт честный статус, а не другой показатель,
подобранный до красивого числа.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.methods.features import (
    FeatureConfig,
    build_features,
    feature_names,
    sample_pixels,
    sample_pixels_uniform,
)


def stable_hash(text: str) -> int:
    """Детерминированный хеш строки, не зависящий от PYTHONHASHSEED."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


@dataclass
class MainModelConfig:
    """Гиперпараметры. Подбираются на validation, к тесту применяются без изменений."""

    n_estimators: int = 400
    learning_rate: float = 0.06
    num_leaves: int = 63
    min_child_samples: int = 80
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    ensemble_size: int = 3
    pixels_per_chip: int = 4000
    #: Для калибровки берём больше пикселей: целевой класс редкий, нужна статистика.
    calibration_pixels: int = 20_000
    positive_share: float = 0.5
    #: Выравнивать ли вклад событий. При выборке по чипам крупные события забирают
    #: большую часть пикселей: USA и Paraguay дают 136 чипов из 257, то есть больше
    #: половины обучающего материала, и модель настраивается на их условия съёмки.
    #: При выравнивании каждое событие получает одинаковый пиксельный бюджет.
    event_balanced: bool = True
    seed: int = 42
    features: FeatureConfig = field(default_factory=FeatureConfig)
    target_name: str = "временное затопление"

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


class MainModel:
    """Ансамбль LightGBM плюс изотоническая калибровка."""

    def __init__(self, config: MainModelConfig | None = None) -> None:
        self.config = config or MainModelConfig()
        self.boosters: list = []
        self.calibrator = None
        self.feature_names: list[str] = feature_names(self.config.features)
        self.threshold: float | None = None
        self.trained_on: list[str] = []
        self.calibrated_on: list[str] = []

    # ── обучение ─────────────────────────────────────────────────────────────

    def fit(self, samples: Iterable[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> "MainModel":
        """samples: (chip_id, vv, vh, target, valid) — ТОЛЬКО обучающая часть.

        Ансамбль строится на разных подвыборках пикселей с разными зёрнами: это и
        даёт разброс, по которому потом считается неопределённость.
        """
        import lightgbm as lgb

        rows: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        chips: list[str] = []
        for chip_id, vv, vh, target, valid in samples:
            cube = build_features(vv, vh, self.config.features)
            # Встроенный hash() строк рандомизируется на каждый процесс, поэтому
            # зерно бралось бы разное при каждом запуске и обучение переставало бы
            # воспроизводиться. Берём устойчивый хеш.
            rng = np.random.default_rng(self.config.seed + stable_hash(chip_id) % 10_000)
            x, y = sample_pixels(
                cube, target, valid, self.config.pixels_per_chip, rng, self.config.positive_share
            )
            if x.size:
                rows.append(x)
                labels.append(y)
                chips.append(chip_id)

        if not rows:
            raise ValueError("обучающая выборка пуста")

        if self.config.event_balanced:
            rows, labels = self._balance_by_event(chips, rows, labels)

        features = np.concatenate(rows).astype(np.float32)
        answers = np.concatenate(labels).astype(np.int8)
        self.trained_on = chips
        self.boosters = []

        for index in range(self.config.ensemble_size):
            seed = self.config.seed + index
            rng = np.random.default_rng(seed)
            # Бэггинг по строкам: каждая модель видит свои 80 % пикселей.
            take = rng.choice(len(features), size=int(0.8 * len(features)), replace=False)
            booster = lgb.LGBMClassifier(
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
            # feature_name не передаём: sklearn-обёртка потом ругается на безымянные
            # массивы при предсказании, а имена нам нужны только для отчёта.
            booster.fit(features[take], answers[take])
            self.boosters.append(booster)
        return self

    def _balance_by_event(
        self, chips: list[str], rows: list[np.ndarray], labels: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Выравнивает пиксельный вклад событий.

        Событие — это одна съёмочная кампания со своим углом наблюдения, своим типом
        местности и своей статистикой обратного рассеяния. Если одно событие даёт
        половину обучающих пикселей, модель настраивается на его условия, а на
        остальных теряет. Поэтому бюджет пикселей делится между событиями поровну,
        а внутри события — между его чипами.
        """
        from collections import defaultdict

        by_event: dict[str, list[int]] = defaultdict(list)
        for index, chip_id in enumerate(chips):
            by_event[chip_id.split("_")[0]].append(index)

        if len(by_event) < 2:
            return rows, labels

        budget = min(sum(len(rows[i]) for i in idx) for idx in by_event.values())
        rng = np.random.default_rng(self.config.seed + 991)
        new_rows: list[np.ndarray] = []
        new_labels: list[np.ndarray] = []
        for indexes in by_event.values():
            per_chip = max(1, budget // len(indexes))
            for index in indexes:
                take = min(per_chip, len(rows[index]))
                picked = rng.choice(len(rows[index]), size=take, replace=False)
                new_rows.append(rows[index][picked])
                new_labels.append(labels[index][picked])
        return new_rows, new_labels

    def calibrate(
        self, samples: Iterable[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> dict[str, float]:
        """Изотоническая калибровка на validation. Тест при этом не используется.

        Выборка здесь равномерная, с природными долями классов. Обучение идёт на
        сбалансированных пикселях, иначе редкий класс утонул бы; но если калибровать
        на той же сбалансированной выборке, изотоническая регрессия выучит
        искусственную долю воды в половину кадра, и значение 0,7 перестанет означать
        наблюдаемую долю 0,7 на настоящем снимке, где воды считаные проценты.
        """
        from sklearn.isotonic import IsotonicRegression

        raw: list[np.ndarray] = []
        answers: list[np.ndarray] = []
        chips: list[str] = []
        rng = np.random.default_rng(self.config.seed + 777)
        for chip_id, vv, vh, target, valid in samples:
            cube = build_features(vv, vh, self.config.features)
            x, y = sample_pixels_uniform(cube, target, valid, self.config.calibration_pixels, rng)
            if not x.size:
                continue
            raw.append(self._raw_mean(x))
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

    def _raw_mean(self, features: np.ndarray) -> np.ndarray:
        """Средняя сырая вероятность по ансамблю."""
        stacked = np.stack([b.predict_proba(features)[:, 1] for b in self.boosters])
        return stacked.mean(axis=0)

    def _raw_stack(self, features: np.ndarray) -> np.ndarray:
        return np.stack([b.predict_proba(features)[:, 1] for b in self.boosters])

    def predict_chip(self, vv: np.ndarray, vh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Вероятность временного затопления и неопределённость, оба (H, W) float32.

        Работает по одному Sentinel-1 и ничему больше: никаких дополнительных слоёв
        на применении не требуется.
        """
        if not self.boosters:
            raise RuntimeError("модель не обучена")
        cube = build_features(vv, vh, self.config.features)
        height, width, n_features = cube.shape
        flat = cube.reshape(-1, n_features)

        stack = self._raw_stack(flat)
        raw = stack.mean(axis=0)
        spread = stack.std(axis=0) if len(self.boosters) > 1 else np.zeros_like(raw)

        prob = self.calibrator.predict(raw) if self.calibrator is not None else raw
        prob = np.clip(prob, 0.0, 1.0).astype(np.float32)

        # Энтропия нормирована к единице: 0 — полная определённость, 1 — граница 0,5.
        eps = 1e-6
        p = np.clip(prob, eps, 1 - eps)
        entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        uncertainty = np.clip(0.7 * entropy + 0.3 * (spread / 0.5), 0.0, 1.0).astype(np.float32)

        return prob.reshape(height, width), uncertainty.reshape(height, width)

    def predict_mask(self, prob: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Бинарная маска строго по правилу mask = 1 ⟺ p >= threshold."""
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
                {"boosters": self.boosters, "calibrator": self.calibrator}, handle, protocol=5
            )
        meta = {
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
    def load(cls, directory: str | Path) -> "MainModel":
        directory = Path(directory)
        meta = json.loads((directory / "main_model.json").read_text(encoding="utf-8"))
        config_payload = dict(meta["config"])
        features_payload = config_payload.pop("features", {})
        config = MainModelConfig(
            **config_payload,
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
            payload = pickle.load(handle)
        model.boosters = payload["boosters"]
        model.calibrator = payload["calibrator"]
        model.feature_names = meta["feature_names"]
        model.threshold = meta.get("threshold")
        model.trained_on = meta.get("trained_on", [])
        model.calibrated_on = meta.get("calibrated_on", [])
        return model

    def feature_importance(self) -> list[tuple[str, float]]:
        """Важность признаков, усреднённая по ансамблю — для отчёта и защиты."""
        if not self.boosters:
            return []
        total = np.mean([b.feature_importances_ for b in self.boosters], axis=0)
        pairs = list(zip(self.feature_names, (total / total.sum() if total.sum() else total)))
        return sorted(pairs, key=lambda item: -item[1])
