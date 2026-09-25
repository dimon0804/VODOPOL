"""HTTP-слой панели оператора FloodValue.

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

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.api import config
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


def _require_ctx(state: RunState) -> RunContext:
    if state.ctx is None:
        raise HTTPException(
            status_code=404,
            detail=state.message or "Комплект запуска не загружен.",
        )
    return state.ctx


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
        allow_methods=["GET", "OPTIONS"],
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

    # ── данные ───────────────────────────────────────────────────────────────

    @app.get("/api/run", summary="Паспорт запуска")
    def run_summary() -> JSONResponse:
        ctx = _require_ctx(app.state.run)
        summary = dict(_guard("Паспорт запуска", ctx.summary))
        # Границы дублируем рядом с растрами: панели удобнее брать их одним запросом.
        summary.setdefault("bounds", None)
        summary["raster_bounds"] = summary.get("bounds")
        return json_ok(summary)

    @app.get("/api/assets", summary="Объекты портфеля с ущербом")
    def assets() -> JSONResponse:
        ctx = _require_ctx(app.state.run)
        return json_ok(_guard("Объекты портфеля", ctx.assets))

    @app.get("/api/candidates", summary="Каталог зон дополнительной съёмки")
    def candidates() -> JSONResponse:
        ctx = _require_ctx(app.state.run)
        return json_ok(_guard("Каталог зон", ctx.candidates))

    @app.get("/api/strategies", summary="Сравнение стратегий A/B/C")
    def strategies(
        budget: float | None = Query(
            default=None,
            ge=0,
            description="Бюджет в рублях. Без параметра — бюджет, объявленный в комплекте.",
        ),
    ) -> JSONResponse:
        ctx = _require_ctx(app.state.run)
        return json_ok(_guard("Стратегии A/B/C", lambda: ctx.strategies(budget)))

    @app.get("/api/sensitivity", summary="Таблица чувствительности")
    def sensitivity() -> JSONResponse:
        ctx = _require_ctx(app.state.run)
        return json_ok(_guard("Чувствительность", ctx.sensitivity))

    # ── карта ────────────────────────────────────────────────────────────────

    @app.get("/api/raster/{kind}.png", summary="Слой карты в PNG")
    def raster(kind: str) -> Response:
        run: RunState = app.state.run
        ctx = _require_ctx(run)
        if kind not in config.RASTER_KINDS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Неизвестный слой карты: {kind!r}. "
                    f"Допустимые значения: {', '.join(config.RASTER_KINDS)}."
                ),
            )
        cached = run.raster_cache.get(kind)
        if cached is None:
            try:
                png, bounds = _guard(f"Слой {kind}", lambda: ctx.raster_png(kind))
            except HTTPException as exc:
                if kind == "s1":
                    # Известный дефект точки входа: путь s1_path лежит в
                    # run_metadata.json, но RunContext.summary() его не отдаёт,
                    # поэтому подложку построить нечем. Панель работает без неё.
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "Подложка S1 недоступна: паспорт запуска не отдаёт путь "
                            "к исходному чипу S1Hand. Карта работает без радарной "
                            "подложки, остальные слои не затронуты. "
                            f"Ответ расчёта: {exc.detail}"
                        ),
                    ) from exc
                raise
            cached = (png, [float(value) for value in bounds])
            if len(run.raster_cache) >= config.RASTER_CACHE_SIZE:
                run.raster_cache.clear()
            run.raster_cache[kind] = cached
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
    def bundle() -> Response:
        run: RunState = app.state.run
        ctx = _require_ctx(run)
        archive = _guard("Выгрузка комплекта", ctx.bundle_zip)
        run_id = "demo"
        try:
            run_id = str(ctx.summary().get("run_id") or "run")
        except Exception:  # pragma: no cover — имя файла не повод ронять выгрузку
            pass
        name = f"floodvalue_{run_id}.zip"
        return Response(
            content=archive,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{name}"',
                "X-Run-Id": run_id,
            },
        )

    # ── статика панели ───────────────────────────────────────────────────────

    if config.WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")

    return app


app = create_app()
