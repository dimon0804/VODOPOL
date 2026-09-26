# Панель оператора «Водополь».
# Исходный набор Sen1Floods11 в образ не кладём: он качается отдельной командой
# и монтируется снаружи. Веса модели и готовые комплекты тоже монтируются, чтобы
# пересборка образа не требовалась при новом запуске.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# GDAL и PROJ нужны rasterio и pyproj в бинарных колёсах уже зашиты,
# но libexpat и libgomp требуются LightGBM и rasterio в рантайме.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости отдельным слоем: правка кода не пересобирает колёса.
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

COPY src/ src/
COPY web/ web/
COPY splits/ splits/

# Точки монтирования создаём заранее. Без них discover_run_dir не находит каталог
# outputs и панель поднимается пустой, если тома почему-то не подключились —
# вместо внятного «комплект не найден» получается загадочный экран.
RUN mkdir -p outputs models data/cache

EXPOSE 8000
ENV FLOODVALUE_HOST=0.0.0.0 FLOODVALUE_PORT=8000

# Пишет сервис ровно в один каталог — outputs, куда складывается собранный из
# панели комплект. В образ и в остальные тома не пишет ничего. Работает под
# непривилегированным пользователем — на публичном домене это не роскошь.
RUN useradd --create-home --uid 10001 vodopol && chown -R vodopol:vodopol /app
USER vodopol

# Проверка живости средствами stdlib: curl в образ ради этого не тащим.
# docker compose ps и deploy/deploy.sh опираются на этот статус.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "src.api"]
