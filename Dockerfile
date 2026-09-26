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

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY web/ web/
COPY splits/ splits/

EXPOSE 8000
ENV FLOODVALUE_HOST=0.0.0.0 FLOODVALUE_PORT=8000

CMD ["python", "-m", "src.api"]
