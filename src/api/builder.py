"""Сборка комплекта по запросу из панели.

Панель умеет не только переключаться между готовыми комплектами, но и собрать
новый: оператор выбирает чип и бюджет, сервис запускает тот же самый
``python -m src.cli.run_bundle``, что и человек в терминале.

Почему именно подпроцесс, а не вызов функции внутри сервиса:

* прогон считает минуту-полторы и занимает память под LightGBM — падение или
  переполнение внутри него не должно ронять веб-сервис;
* в терминале и из панели выполняется буквально одна и та же команда, поэтому
  результат совпадает по построению, а не по обещанию;
* вывод команды читается построчно и отдаётся в панель как есть — оператор видит
  тот же протокол, что увидел бы в консоли, включая строку валидатора.

Одновременно выполняется не больше одной сборки. Это не упрощение ради простоты:
два параллельных прогона на ноутбуке докладчика — это своп и сорванная демонстрация.
"""

from __future__ import annotations

import itertools
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.api import config

#: Идентификатор чипа в наборе: Событие_номер. Дефис допустим (Sri-Lanka).
CHIP_ID_RE = re.compile(r"^[A-Za-z][A-Za-z-]*_\d+$")

#: Границы бюджета. Ноль — осмысленный сценарий (стратегия A), потолок защищает
#: от опечатки в десять лишних нулей, а не от злого умысла.
BUDGET_MIN = 0.0
BUDGET_MAX = 1_000_000_000.0

#: Сколько последних строк вывода держим. Полный протокол короткий, но ограничение
#: нужно: зацикленный вывод не должен съесть память сервиса.
LOG_LIMIT = 400

#: Потолок времени на прогон. Больше — значит что-то пошло не так, и лучше
#: сказать об этом, чем держать панель в «идёт сборка» до конца света.
TIMEOUT_SEC = 1800


def cache_root() -> Path:
    """Каталог, куда fetch_data складывает слои набора."""
    return config.PROJECT_ROOT / "data" / "cache"


def chip_layers_missing(chip_id: str) -> list[str]:
    """Слои чипа, которых ещё нет на диске."""
    from src.contracts import HAND_LAYERS
    from src.data import fetch

    return [
        layer
        for layer in HAND_LAYERS
        if not fetch.layer_path(chip_id, layer, cache_root()).is_file()
    ]


def available_chips() -> list[dict[str, Any]]:
    """Все чипы набора, которые можно собрать.

    Раньше сюда попадали только выкачанные чипы, и для любого другого оператору
    предлагали идти в терминал за fetch_data. Теперь список — весь сплит, а
    недостающие слои одного чипа сборка скачивает сама перед прогоном: это
    несколько мегабайт из официального бакета, а не 700 МБ всего набора.
    """
    parts = config.event_parts()
    built = {}
    for path in config.list_run_dirs():
        # Имя каталога запуска: ГГГГММДД-<chip_id>-b<бюджет>. Чип вынимаем из него,
        # чтобы не открывать сотню паспортов ради одного поля.
        chunks = path.name.split("-", 1)
        if len(chunks) == 2:
            body = chunks[1].rsplit("-b", 1)[0]
            built[body] = path.name

    s1_dir = cache_root() / "S1Hand"
    items: list[dict[str, Any]] = []
    for chip_id, event in config.all_split_chips():
        items.append(
            {
                "chip_id": chip_id,
                "event_id": event,
                "part": parts.get(event, ""),
                "run_id": built.get(chip_id, ""),
                "downloaded": (s1_dir / f"{chip_id}_S1Hand.tif").is_file(),
            }
        )
    items.sort(key=lambda item: (item["event_id"], item["chip_id"]))
    return items


def validate_chip(chip_id: str) -> str:
    """Проверяет чип по каталогу на диске и возвращает его канонический идентификатор.

    Значение приходит из запроса, поэтому проверяется дважды: формой и наличием
    в каталоге. Собрать что-то, кроме перечисленных чипов, этой ручкой нельзя.
    """
    name = (chip_id or "").strip()
    if not CHIP_ID_RE.match(name):
        raise ValueError(
            f"Идентификатор чипа {chip_id!r} не похож на чип Sen1Floods11: "
            "ожидается вид India_900498."
        )
    known = {chip for chip, _event in config.all_split_chips()}
    if name not in known:
        raise ValueError(
            f"Чипа {name} нет в размеченной части Sen1Floods11: собрать можно только "
            "чип из списков сплита."
        )
    return name


def validate_budget(budget: Any) -> float:
    try:
        value = float(budget)
    except (TypeError, ValueError):
        raise ValueError(f"Бюджет {budget!r} не число.") from None
    if not (BUDGET_MIN <= value <= BUDGET_MAX):
        raise ValueError(
            f"Бюджет вне допустимого диапазона: ожидается от {BUDGET_MIN:,.0f} "
            f"до {BUDGET_MAX:,.0f} руб."
        )
    return value


@dataclass
class BuildJob:
    """Одна сборка: что просили, как идёт, что получилось."""

    job_id: str
    chip_id: str
    budget: float
    #: running | done | error
    status: str = "running"
    lines: list[str] = field(default_factory=list)
    run_id: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "chip_id": self.chip_id,
            "budget_rub": self.budget,
            "status": self.status,
            "lines": list(self.lines),
            "run_id": self.run_id,
            "error": self.error,
            "elapsed_sec": round((self.finished_at or time.time()) - self.started_at, 1),
        }


class BuildManager:
    """Очередь на одну сборку и журнал того, что уже собирали."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, BuildJob] = {}
        self._active: str = ""

    # ── чтение ───────────────────────────────────────────────────────────────

    def get(self, job_id: str) -> BuildJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> BuildJob | None:
        with self._lock:
            return self._jobs.get(self._active) if self._active else None

    # ── запуск ───────────────────────────────────────────────────────────────

    def start(self, chip_id: str, budget: float) -> BuildJob:
        """Ставит сборку, если свободно. Иначе объясняет, чем сервис занят."""
        chip_id = validate_chip(chip_id)
        budget = validate_budget(budget)
        with self._lock:
            running = self._jobs.get(self._active) if self._active else None
            if running is not None and running.status == "running":
                raise RuntimeError(
                    f"Уже идёт сборка чипа {running.chip_id}. Дождитесь её окончания: "
                    "два прогона одновременно не ускорят работу, а займут память."
                )
            job = BuildJob(job_id=uuid.uuid4().hex[:12], chip_id=chip_id, budget=budget)
            self._jobs[job.job_id] = job
            self._active = job.job_id
            self._forget_old()
        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def _forget_old(self) -> None:
        """Держим последние два десятка сборок: журнал, а не архив."""
        if len(self._jobs) <= 20:
            return
        stale = sorted(self._jobs.values(), key=lambda j: j.started_at)[:-20]
        for job in stale:
            if job.job_id != self._active:
                self._jobs.pop(job.job_id, None)

    # ── выполнение ───────────────────────────────────────────────────────────

    def _log(self, job: BuildJob, text: str) -> None:
        with self._lock:
            job.lines.append(text)
            if len(job.lines) > LOG_LIMIT:
                del job.lines[: len(job.lines) - LOG_LIMIT]

    def _ensure_chip(self, job: BuildJob) -> bool:
        """Докачивает слои чипа и метаданные набора, если их нет на диске.

        Это ровно то, что fetch_data сделал бы для одного чипа: те же файлы из того
        же официального бакета в тот же data/cache. Возвращает False, если скачать
        не удалось, — тогда сборку не запускаем, а говорим почему.
        """
        from src.data import fetch

        missing = chip_layers_missing(job.chip_id)
        meta = cache_root() / "Sen1Floods11_Metadata.geojson"
        if not missing and meta.is_file():
            return True
        root = cache_root()
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError:
            self._finish(
                job,
                "error",
                "чип не выкачан, а каталог data/cache недоступен на запись — скачать его некуда",
            )
            return False
        if missing:
            self._log(job, f"Чип {job.chip_id} не выкачан: скачиваю {len(missing)} слоя из официального бакета Sen1Floods11…")
        try:
            for layer in missing:
                path = fetch.fetch_one(job.chip_id, layer, root)
                self._log(job, f"  {layer}: {path.stat().st_size / 1e6:.1f} МБ")
            if not meta.is_file():
                fetch.fetch_metadata(root)
                self._log(job, "  Sen1Floods11_Metadata.geojson: даты съёмки событий")
        except Exception as exc:  # сеть, бакет, диск — всё одинаково честно сказать
            self._finish(job, "error", f"не удалось скачать чип {job.chip_id}: {exc}. Проверьте интернет и повторите.")
            return False
        self._log(job, "Слои на месте, запускаю сборку комплекта.")
        return True

    def _run(self, job: BuildJob) -> None:
        if not self._ensure_chip(job):
            return
        command = [
            sys.executable,
            "-X",
            "utf8",
            "-u",  # без буферизации: строки должны появляться в панели по ходу дела
            "-m",
            "src.cli.run_bundle",
            "--chip",
            job.chip_id,
            "--budget",
            f"{job.budget:.2f}",
        ]
        try:
            # shell=False и список аргументов: chip_id уже проверен по каталогу, но
            # передавать его через строку команды всё равно незачем.
            process = subprocess.Popen(
                command,
                cwd=str(config.PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            self._finish(job, "error", f"не удалось запустить сборку: {exc}")
            return

        deadline = time.time() + TIMEOUT_SEC
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()
            if text:
                with self._lock:
                    job.lines.append(text)
                    if len(job.lines) > LOG_LIMIT:
                        del job.lines[: len(job.lines) - LOG_LIMIT]
            if time.time() > deadline:
                process.kill()
                self._finish(job, "error", f"сборка не уложилась в {TIMEOUT_SEC // 60} минут")
                return

        code = process.wait()
        if code != 0:
            tail = " ".join(job.lines[-3:]) or f"код возврата {code}"
            self._finish(job, "error", f"сборка прервалась: {tail}")
            return

        run_id = self._run_id_from(job)
        if not run_id:
            self._finish(job, "error", "сборка завершилась, но каталог комплекта не найден")
            return
        with self._lock:
            job.run_id = run_id
        self._finish(job, "done", "")

    def _run_id_from(self, job: BuildJob) -> str:
        """Ищет каталог, который только что появился под этот чип.

        Имя запуска строит сам run_bundle, и повторять его формулу здесь значило бы
        завести второй источник правды. Поэтому смотрим на каталог: берём самый
        свежий из тех, что относятся к нужному чипу.
        """
        prefix_hits = [
            path
            for path in config.list_run_dirs()
            if f"-{job.chip_id}-b" in path.name
        ]
        return prefix_hits[0].name if prefix_hits else ""

    def _finish(self, job: BuildJob, status: str, error: str) -> None:
        with self._lock:
            job.status = status
            job.error = error
            job.finished_at = time.time()
            if self._active == job.job_id:
                self._active = ""


#: Один менеджер на процесс: сборка — свойство машины, а не запроса.
MANAGER = BuildManager()


def writable_outputs() -> tuple[bool, str]:
    """Можно ли вообще собирать здесь комплекты.

    В контейнере каталог outputs может быть смонтирован только на чтение — тогда
    кнопка сборки обязана честно сказать об этом заранее, а не падать на середине
    прогона с невнятной ошибкой прав.
    """
    outputs = config.OUTPUTS_DIR
    try:
        outputs.mkdir(parents=True, exist_ok=True)
        probe = outputs / f".write-probe-{next(_PROBE)}"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return False, (
            f"Каталог {outputs} недоступен для записи ({exc.strerror or exc}). "
            "Сборка из панели выключена: в docker-compose.yml том outputs "
            "подключён только на чтение."
        )
    return True, ""


_PROBE = itertools.count()
