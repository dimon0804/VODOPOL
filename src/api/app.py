"""HTTP-слой панели оператора Водополь.

Слой ничего не считает. Каждый эндпоинт — это вызов одного метода
`src.runtime.RunContext` и сериализация того, что вернулось. Так цифры на экране
совпадают с файлами сданного комплекта по построению, а не по договорённости.

Какой комплект открыт:
  1. путь из переменной окружения ``FLOODVALUE_RUN``;
  2. если её нет — самый свежий каталог в ``outputs/`` с ``run_metadata.json``;
  3. если и его нет — ``RunContext.demo()`` (синтетика, помечена ``demo: true``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.api import builder, config
from src.runtime import RunContext

# ─────────────────────────────────────────────────────────────────────────────
# Сериализация
# ─────────────────────────────────────────────────────────────────────────────


def jsonable(value: Any) -> Any:
    """Приводит то, что отдал расчёт, к типам JSON без потери смысла.

    Decimal превращается в число, пустое остаётся пустым (``null``), NaN и Inf
    в выдачу не попадают: по контракту пустое значение — это отсутствие оценки,
    а не ноль.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Decimal):
        return jsonable(float(value))
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    # numpy-скаляры и прочее, что умеет .item()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return jsonable(item())
        except Exception:  # pragma: no cover — защитный путь
            pass
    return str(value)


def json_ok(payload: Any) -> JSONResponse:
    return JSONResponse(content=jsonable(payload))


# ─────────────────────────────────────────────────────────────────────────────
# Состояние сервиса
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RunState:
    """Что именно открыто сейчас и почему."""

    ctx: RunContext | None = None
    run_dir: Path | None = None
    #: env | auto | demo | none
    source: str = "none"
    message: str = ""
    raster_cache: dict[str, tuple[bytes, list[float]]] = field(default_factory=dict)
    #: Открытые по запросу соседние комплекты: run_id -> контекст.
    opened: dict[str, RunContext] = field(default_factory=dict)

    @property
    def is_demo(self) -> bool:
        return bool(self.ctx and self.ctx.is_demo)

    @property
    def status(self) -> str:
        if self.ctx is None:
            return "error"
        return "demo" if self.is_demo else "ok"


def load_state() -> RunState:
    """Открывает комплект по правилам из config, не падая на отсутствии данных."""
    explicit = config.env_run_dir()
    if explicit is not None:
        try:
            ctx = RunContext.load(explicit)
        except FileNotFoundError as exc:
            return RunState(
                ctx=None,
                run_dir=explicit,
                source="env",
                message=(
                    f"Переменная {config.ENV_RUN} указывает на {explicit}, "
                    f"но комплект там не найден. {exc}"
                ),
            )
        return RunState(
            ctx=ctx,
            run_dir=explicit,
            source="env",
            message=f"Комплект открыт по {config.ENV_RUN}.",
        )

    found = config.discover_run_dir()
    if found is not None:
        return RunState(
            ctx=RunContext.load(found),
            run_dir=found,
            source="auto",
            message=f"Комплект найден автоматически в {config.OUTPUTS_DIR.name}/.",
        )

    return RunState(
        ctx=RunContext.demo(),
        run_dir=None,
        source="demo",
        message=(
            "Готовый комплект не найден, открыта демонстрационная синтетика. "
            "Соберите комплект командой python -m src.cli.run_bundle "
            f"или укажите путь в {config.ENV_RUN}."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательное
# ─────────────────────────────────────────────────────────────────────────────


def _run_id_of(ctx: RunContext) -> str:
    """Идентификатор комплекта, не роняя запрос на нечитаемом паспорте."""
    try:
        return str(ctx.summary().get("run_id") or "")
    except Exception:  # pragma: no cover — защитный путь
        return ""


def _select_ctx(state: RunState, run_id: str | None) -> RunContext:
    """Комплект, с которым работает запрос: открытый по умолчанию или выбранный.

    Панель переключает чип параметром ``run``. Контекст соседнего комплекта
    открывается один раз и запоминается: переключение туда-обратно не перечитывает
    файлы. Модель при этом не трогается вообще — читаются готовые артефакты.
    """
    if not run_id:
        return _require_ctx(state)
    current = state.ctx
    if current is not None and not current.is_demo and _run_id_of(current) == run_id:
        return current
    opened = state.opened.get(run_id)
    if opened is not None:
        return opened
    path = config.run_dir_by_id(run_id)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Комплект {run_id!r} не найден в {config.OUTPUTS_DIR.name}/. "
                "Соберите его командой python -m src.cli.run_bundle --chip <id>."
            ),
        )
    ctx = _guard(f"Комплект {run_id}", lambda: RunContext.load(path))
    state.opened[run_id] = ctx
    return ctx


def _require_ctx(state: RunState) -> RunContext:
    if state.ctx is None:
        raise HTTPException(
            status_code=404,
            detail=state.message or "Комплект запуска не загружен.",
        )
    return state.ctx


def _event_footprints(app: FastAPI) -> tuple[dict[str, Any], str]:
    """Контуры событий Sen1Floods11 для обзорной карты.

    Сначала файл, который скачала fetch_data. Если набор не выкачан — один раз
    берём тот же файл из официального бакета и держим в памяти: в репозиторий
    чужие данные не кладём. Нет сети — обзор показывает только собранные чипы.
    """
    cached = getattr(app.state, "event_footprints", None)
    if cached is not None:
        return cached
    empty = {"type": "FeatureCollection", "features": []}
    result: tuple[dict[str, Any], str]
    path = config.PROJECT_ROOT / config.METADATA_PATH
    try:
        if path.is_file():
            result = (json.loads(path.read_text(encoding="utf-8")), "data/cache")
        else:
            import urllib.request

            with urllib.request.urlopen(config.METADATA_URL, timeout=6) as resp:
                result = (json.loads(resp.read().decode("utf-8")), "официальный бакет Sen1Floods11")
    except Exception:
        # Не кэшируем провал: сеть может появиться, а файл — выкачаться позже.
        return empty, "недоступно: набор не выкачан и бакет не ответил"
    app.state.event_footprints = result
    return result


def _read_chip_bounds(chip_ids: list[str]) -> dict[str, list[float] | None]:
    """Границы снимков чипов: с диска, если выкачан, иначе заголовок из бакета.

    Читается только заголовок GeoTIFF — HTTP-запрос с диапазоном, а не весь файл.
    Чип, который не прочитался (нет сети), остаётся без контура, а не роняет ответ.
    """
    from concurrent.futures import ThreadPoolExecutor

    import rasterio

    env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_TIMEOUT": "15",
    }

    def one(chip: str) -> tuple[str, list[float] | None]:
        local = config.PROJECT_ROOT / config.S1_LOCAL.format(chip=chip)
        source = str(local) if local.is_file() else "/vsicurl/" + config.S1_REMOTE.format(chip=chip)
        try:
            with rasterio.Env(**env), rasterio.open(source) as ds:
                return chip, [float(v) for v in ds.bounds]
        except Exception:
            return chip, None

    with ThreadPoolExecutor(max_workers=16) as pool:
        return dict(pool.map(one, chip_ids))


def _guard(action: str, call: Callable[[], Any]) -> Any:
    """Ошибку расчёта превращает в понятный ответ, а не в трейсбек на экране."""
    try:
        return call()
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"{action}: {exc}") from exc
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{action}: комплект неполон или повреждён ({exc}).",
        ) from exc
    except Exception as exc:  # расчёт упал — сообщаем, но сервис не роняем
        raise HTTPException(
            status_code=503,
            detail=f"{action}: расчёт недоступен ({type(exc).__name__}: {exc}).",
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Приложение
# ─────────────────────────────────────────────────────────────────────────────


def create_app(state: RunState | None = None) -> FastAPI:
    app = FastAPI(
        title=config.API_TITLE,
        version=config.API_VERSION,
        description="Тонкий слой над src.runtime.RunContext. Вычислений здесь нет.",
    )
    app.state.run = state if state is not None else load_state()

    # Локальная разработка: панель может открываться с другого порта или из файла.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Bounds", "X-Run-Id", "Content-Disposition"],
    )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc.detail), "detail": str(exc.detail), "path": request.url.path},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def any_error(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Внутренняя ошибка сервиса: {type(exc).__name__}.",
                "detail": str(exc),
                "path": request.url.path,
            },
        )

    # ── состояние ────────────────────────────────────────────────────────────

    @app.get("/api/health", summary="Состояние сервиса")
    def health() -> JSONResponse:
        run: RunState = app.state.run
        payload = {
            "status": run.status,
            "version": config.API_VERSION,
            "source": run.source,
            "run_dir": str(run.run_dir) if run.run_dir else None,
            "demo": run.is_demo,
            "message": run.message,
            "raster_kinds": list(config.RASTER_KINDS),
            "run_id": None,
            "chip_id": None,
        }
        if run.ctx is not None:
            try:
                summary = run.ctx.summary()
                payload["run_id"] = summary.get("run_id")
                payload["chip_id"] = summary.get("chip_id")
            except Exception as exc:  # комплект есть, но паспорт не читается
                payload["status"] = "error"
                payload["message"] = f"Паспорт запуска не прочитан: {exc}"
        return json_ok(payload)

    @app.get("/api/runs", summary="Собранные комплекты, доступные для показа")
    def runs() -> JSONResponse:
        """Перечень комплектов для выпадающего списка панели.

        Собирается из каталога outputs/, а не из конфигурации: собрали новый чип
        командой run_bundle — он появляется в списке без перезапуска сервиса.
        """
        state: RunState = app.state.run
        current = _run_id_of(state.ctx) if state.ctx is not None else ""
        parts = config.event_parts()
        items: list[dict[str, Any]] = []
        for path in config.list_run_dirs():
            try:
                ctx = state.opened.get(path.name) or RunContext.load(path)
            except Exception:  # каталог есть, читать нечего — молча пропускаем
                continue
            try:
                summary = ctx.summary()
            except Exception:
                continue
            event = summary.get("event_id") or ""
            items.append(
                {
                    "run_id": summary.get("run_id") or path.name,
                    "chip_id": summary.get("chip_id"),
                    "event_id": event,
                    "observation_date": summary.get("observation_date"),
                    "budget_rub": summary.get("budget_rub"),
                    "threshold": summary.get("threshold"),
                    "part": parts.get(str(event), ""),
                    # Цель запуска в списке обязательна: два прогона по одному чипу
                    # с разными целями отличаются только ценой, и без подписи они
                    # выглядят как дубль одной строки.
                    "purpose": (summary.get("purpose") or {}).get("key", ""),
                    "purpose_title": (summary.get("purpose") or {}).get("title_ru", ""),
                    "current": (summary.get("run_id") or path.name) == current,
                }
            )
        return json_ok({"current": current, "runs": items})

    @app.get("/api/overview", summary="Все события набора и все собранные чипы — для обзорной карты")
    def overview() -> JSONResponse:
        """Обзорный слой карты: всё сразу, как поля в «Фенологе».

        События — контуры из официального Sen1Floods11_Metadata.geojson (его
        скачивает fetch_data). Чипы — собранные комплекты с границами и ущербом из
        паспорта запуска. Ничего не считается: всё читается как есть.
        """
        state: RunState = app.state.run
        current = _run_id_of(state.ctx) if state.ctx is not None else ""
        parts = config.event_parts()
        chips: list[dict[str, Any]] = []
        for path in config.list_run_dirs():
            try:
                ctx = state.opened.get(path.name) or RunContext.load(path)
                summary = ctx.summary()
            except Exception:
                continue
            event = summary.get("event_id") or ""
            chips.append(
                {
                    "run_id": summary.get("run_id") or path.name,
                    "chip_id": summary.get("chip_id"),
                    "event_id": event,
                    "part": parts.get(str(event), ""),
                    "observation_date": summary.get("observation_date"),
                    "bounds": summary.get("bounds"),
                    "total_expected_loss_rub": summary.get("total_expected_loss_rub"),
                    "current": (summary.get("run_id") or path.name) == current,
                }
            )
        events, source = _event_footprints(app)
        # Имя события в метаданных сводим к имени в сплите — иначе Mekong на карте
        # назывался бы Cambodia и не открывался бы кликом.
        events = {
            **events,
            "features": [
                {**f, "properties": {**(f.get("properties") or {}),
                                     "event": config.EVENT_ALIASES.get((f.get("properties") or {}).get("location"),
                                                                       (f.get("properties") or {}).get("location"))}}
                for f in events.get("features", [])
            ],
        }
        # По каждому событию: сколько размеченных чипов в наборе и сколько собрано.
        # Обзор мира показывает события, а не отдельные чипы: чип выбирают уже в
        # регионе события, где он виден среди соседей.
        stats: dict[str, dict[str, Any]] = {}
        for chip_id, event in config.all_split_chips():
            item = stats.setdefault(event, {"chips": 0, "built": 0, "part": parts.get(event, "")})
            item["chips"] += 1
        for chip in chips:
            if chip["event_id"] in stats:
                stats[chip["event_id"]]["built"] += 1
        return json_ok(
            {"current": current, "chips": chips, "events": events, "events_source": source, "event_stats": stats}
        )

    @app.get("/api/event-chips", summary="Все чипы одного события с контурами — как поля региона")
    def event_chips(event: str = Query(..., description="Событие Sen1Floods11, например Bolivia")) -> JSONResponse:
        """Регион события целиком: все его чипы из сплита, собранные и нет.

        Контур чипа — границы его снимка S1Hand. Выкачанный файл читается с диска,
        остальные — только заголовок из официального бакета по HTTP (несколько
        килобайт на чип, сам снимок не качается). Результат держится в памяти.
        """
        ids = config.event_chip_ids(event)
        if not ids:
            raise HTTPException(status_code=404, detail=f"события {event} нет в списках сплита")
        cache = getattr(app.state, "event_chip_cache", None)
        if cache is None:
            cache = app.state.event_chip_cache = {}
        if event not in cache:
            cache[event] = _read_chip_bounds(ids)
        bounds = cache[event]

        state: RunState = app.state.run
        current = _run_id_of(state.ctx) if state.ctx is not None else ""
        built: dict[str, str] = {}
        for path in config.list_run_dirs():
            try:
                summary = (state.opened.get(path.name) or RunContext.load(path)).summary()
            except Exception:
                continue
            if summary.get("chip_id"):
                built[str(summary["chip_id"])] = summary.get("run_id") or path.name
        chips = [
            {
                "chip_id": chip,
                "bounds": bounds.get(chip),
                "run_id": built.get(chip),
                "current": bool(built.get(chip)) and built.get(chip) == current,
            }
            for chip in ids
        ]
        return json_ok(
            {
                "event": event,
                "part": config.event_parts().get(event, ""),
                "chips": chips,
                "missing": sum(1 for c in chips if not c["bounds"]),
            }
        )

    # ── сборка нового комплекта ──────────────────────────────────────────────

    @app.get("/api/chips", summary="Чипы, доступные для сборки")
    def chips() -> JSONResponse:
        """Каталог выкачанных чипов плюс ответ на вопрос, можно ли здесь собирать.

        Признак `writable` панель спрашивает заранее: если каталог outputs
        смонтирован только на чтение, кнопку сборки честнее погасить сразу, чем
        уронить прогон на середине непонятной ошибкой прав.
        """
        writable, reason = builder.writable_outputs()
        active = builder.MANAGER.active()
        return json_ok(
            {
                "writable": writable,
                "reason": reason,
                "busy": active.payload() if active is not None else None,
                "chips": builder.available_chips(),
            }
        )

    @app.post("/api/build", summary="Собрать комплект по чипу")
    def build(payload: dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
        """Запускает ту же команду run_bundle, что человек набрал бы в терминале.

        Ответ возвращается сразу, не дожидаясь конца прогона: считает он минуту с
        лишним, и держать соединение всё это время незачем. Панель опрашивает
        /api/build/{job_id} и показывает протокол по мере появления строк.
        """
        writable, reason = builder.writable_outputs()
        if not writable:
            raise HTTPException(status_code=409, detail=reason)
        try:
            job = builder.MANAGER.start(
                str(payload.get("chip_id", "")),
                payload.get("budget_rub", payload.get("budget")),
            )
        except ValueError as exc:  # неверный чип или бюджет — вина запроса
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:  # занято другой сборкой
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return json_ok(job.payload())

    @app.get("/api/build/{job_id}", summary="Ход сборки")
    def build_status(job_id: str) -> JSONResponse:
        job = builder.MANAGER.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"Сборка {job_id!r} не найдена: сервис мог быть перезапущен.",
            )
        return json_ok(job.payload())

    # ── данные ───────────────────────────────────────────────────────────────

    @app.get("/api/run", summary="Паспорт запуска")
    def run_summary(
    run: str | None = Query(
        default=None,
        description="Идентификатор комплекта. Без параметра — открытый по умолчанию.",
    ),
    ) -> JSONResponse:
        ctx = _select_ctx(app.state.run, run)
        summary = dict(_guard("Паспорт запуска", ctx.summary))
        # Границы дублируем рядом с растрами: панели удобнее брать их одним запросом.
        summary.setdefault("bounds", None)
        summary["raster_bounds"] = summary.get("bounds")
        return json_ok(summary)

    @app.get("/api/assets", summary="Объекты портфеля с ущербом")
    def assets(
    run: str | None = Query(
        default=None,
        description="Идентификатор комплекта. Без параметра — открытый по умолчанию.",
    ),
    ) -> JSONResponse:
        ctx = _select_ctx(app.state.run, run)
        return json_ok(_guard("Объекты портфеля", ctx.assets))

    @app.get("/api/candidates", summary="Каталог зон дополнительной съёмки")
    def candidates(
    run: str | None = Query(
        default=None,
        description="Идентификатор комплекта. Без параметра — открытый по умолчанию.",
    ),
    ) -> JSONResponse:
        ctx = _select_ctx(app.state.run, run)
        return json_ok(_guard("Каталог зон", ctx.candidates))

    @app.get("/api/strategies", summary="Сравнение стратегий A/B/C")
    def strategies(
        budget: float | None = Query(
            default=None,
            ge=0,
            description="Бюджет в рублях. Без параметра — бюджет, объявленный в комплекте.",
        ),
    run: str | None = Query(
        default=None,
        description="Идентификатор комплекта. Без параметра — открытый по умолчанию.",
    ),
    ) -> JSONResponse:
        ctx = _select_ctx(app.state.run, run)
        return json_ok(_guard("Стратегии A/B/C", lambda: ctx.strategies(budget)))

    @app.get("/api/sensitivity", summary="Таблица чувствительности")
    def sensitivity(
    run: str | None = Query(
        default=None,
        description="Идентификатор комплекта. Без параметра — открытый по умолчанию.",
    ),
    ) -> JSONResponse:
        ctx = _select_ctx(app.state.run, run)
        return json_ok(_guard("Чувствительность", ctx.sensitivity))

    # ── карта ────────────────────────────────────────────────────────────────

    @app.get("/api/raster/{kind}.png", summary="Слой карты в PNG")
    def raster(
        kind: str,
        run: str | None = Query(
            default=None,
            description="Идентификатор комплекта. Без параметра — открытый по умолчанию.",
        ),
    ) -> Response:
        state: RunState = app.state.run
        ctx = _select_ctx(state, run)
        if kind not in config.RASTER_KINDS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Неизвестный слой карты: {kind!r}. "
                    f"Допустимые значения: {', '.join(config.RASTER_KINDS)}."
                ),
            )
        # Ключ кеша включает комплект: иначе переключение чипа отдавало бы
        # картинку предыдущего, и карта врала бы молча.
        cache_key = (_run_id_of(ctx) or "current") + "|" + kind
        cached = state.raster_cache.get(cache_key)
        if cached is None:
            try:
                png, bounds = _guard(f"Слой {kind}", lambda: ctx.raster_png(kind))
            except HTTPException as exc:
                if kind == "s1":
                    # Исходный чип S1Hand в комплект не входит: он лежит в data/cache
                    # и появляется только после python -m src.cli.fetch_data. Путь к
                    # нему паспорт отдаёт, отсутствует сам файл. Панель работает без
                    # подложки, и оператору надо сказать, как её вернуть.
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "Подложка S1 недоступна: исходный чип S1Hand не выкачан. "
                            "Выполните python -m src.cli.fetch_data — подложка появится "
                            "без перезапуска. Остальные слои карты не затронуты. "
                            f"Ответ расчёта: {exc.detail}"
                        ),
                    ) from exc
                raise
            cached = (png, [float(value) for value in bounds])
            if len(state.raster_cache) >= config.RASTER_CACHE_SIZE:
                state.raster_cache.clear()
            state.raster_cache[cache_key] = cached
        png, bounds = cached
        return Response(
            content=png,
            media_type="image/png",
            headers={
                "X-Bounds": json.dumps(bounds),
                "Cache-Control": "no-cache",
            },
        )

    # ── выгрузка ─────────────────────────────────────────────────────────────

    @app.get("/api/bundle.zip", summary="Комплект запуска одним архивом")
    def bundle(
        run: str | None = Query(
            default=None,
            description="Идентификатор комплекта. Без параметра — открытый по умолчанию.",
        ),
    ) -> Response:
        ctx = _select_ctx(app.state.run, run)
        archive = _guard("Выгрузка комплекта", ctx.bundle_zip)
        run_id = "demo"
        try:
            run_id = str(ctx.summary().get("run_id") or "run")
        except Exception:  # pragma: no cover — имя файла не повод ронять выгрузку
            pass
        name = f"vodopol_{run_id}.zip"  # имя продукта, а не технического пакета
        return Response(
            content=archive,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{name}"',
                "X-Run-Id": run_id,
            },
        )

    # ── отдельные файлы комплекта и отчёты модели ────────────────────────────
    #
    # Обе ручки ничего не считают: отдают готовые файлы байт в байт. Список имён
    # закрыт — путь собирается только из белого списка, выйти за каталог нельзя.

    @app.get("/api/files/{name}", summary="Один файл комплекта как есть")
    def run_file(
        name: str,
        run: str | None = Query(default=None, description="Идентификатор комплекта."),
    ) -> Response:
        if name not in config.RUN_FILES:
            raise HTTPException(status_code=404, detail=f"файла {name} в списке выгружаемых нет")
        ctx = _select_ctx(app.state.run, run)
        path = Path(ctx.run_dir) / name
        if ctx.is_demo or not path.is_file():
            raise HTTPException(status_code=404, detail=f"в комплекте нет файла {name}")
        return Response(
            content=path.read_bytes(),
            media_type=config.RUN_FILES[name],
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @app.get("/api/reports/{name}", summary="Отчёт по качеству модели как есть")
    def report_file(name: str) -> Response:
        rel = config.REPORT_FILES.get(name)
        if rel is None:
            raise HTTPException(status_code=404, detail=f"отчёта {name} в списке нет")
        path = config.PROJECT_ROOT / rel
        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"отчёт {rel} не найден: в Docker каталог reports/ монтируется томом",
            )
        return Response(content=path.read_bytes(), media_type="application/json")

    # ── статика панели ───────────────────────────────────────────────────────

    if config.WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")

    return app


app = create_app()
