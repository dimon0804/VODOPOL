#!/usr/bin/env bash
# Развёртывание «Водополя» на сервере.
#
#   ssh dmitry@188.187.214.35
#   git clone https://github.com/dimon0804/VODOPOL.git ~/vodopol
#   cd ~/vodopol && bash deploy/deploy.sh
#
# Скрипт идемпотентен: повторный запуск пересобирает образ и перезапускает
# контейнер. Комплекты запусков и веса при этом не пересобираются — они
# монтируются из каталога репозитория и переживают пересоздание контейнера.
#
# В отличие от «Фенолога», состояния у сервиса нет: панель только читает готовый
# комплект. Поэтому именованный том не нужен, а потеря контейнера ничего не стоит.
set -euo pipefail

IMAGE=vodopol:prod
NAME=vodopol
PORT=8020                 # локальный порт; наружу отдаёт nginx
DOMAIN=vodopol.clv-digital.tech

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "==> Проверяю, что есть что показывать"
if ! ls outputs/*/run_metadata.json >/dev/null 2>&1; then
    echo "В outputs/ нет ни одного комплекта запуска." >&2
    echo "Соберите его: python -m src.cli.run_bundle --chip India_900498 --budget 20000" >&2
    exit 1
fi
echo "    комплектов найдено: $(ls -d outputs/*/ | wc -l)"

echo "==> Собираю образ"
docker build -t "$IMAGE" .

echo "==> Останавливаю прежний контейнер, если он есть"
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "==> Запускаю"
# Слушаем только на локальном адресе: наружу отдаёт nginx, у приложения нет
# забот про TLS. Оба тома в режиме только чтение — панель ничего не пишет,
# и прострелить себе комплект сданного запуска из контейнера невозможно.
#
# FLOODVALUE_RUN не задаём: сервис сам берёт самый свежий комплект из outputs/.
# Чтобы закрепить конкретный, передайте его при запуске скрипта:
#   FLOODVALUE_RUN=outputs/20260926-India_900498-b6000 bash deploy/deploy.sh
#
# data/cache монтируется ради подложки S1: исходный чип S1Hand в комплект не входит,
# и без этого тома слой карты отдаёт 503. Если датасет на сервере не выкачан, каталог
# просто пустой — панель работает, подложки нет, остальные слои не затронуты.
mkdir -p "${ROOT}/data/cache"
docker run -d \
    --name "$NAME" \
    --restart unless-stopped \
    -p 127.0.0.1:${PORT}:8000 \
    -v "${ROOT}/outputs":/app/outputs:ro \
    -v "${ROOT}/models":/app/models:ro \
    -v "${ROOT}/reports":/app/reports:ro \
    -v "${ROOT}/data/cache":/app/data/cache:ro \
    -e FLOODVALUE_HOST=0.0.0.0 \
    -e FLOODVALUE_PORT=8000 \
    -e FLOODVALUE_RUN="${FLOODVALUE_RUN:-}" \
    "$IMAGE"

echo "==> Жду, пока поднимется"
for attempt in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
        echo "    поднялся с ${attempt}-й попытки"
        break
    fi
    sleep 1
    if [ "$attempt" = 30 ]; then
        echo "Сервис не ответил за 30 секунд. Логи:" >&2
        docker logs --tail 40 "$NAME" >&2
        exit 1
    fi
done

echo "==> Что загрузилось"
curl -fsS "http://127.0.0.1:${PORT}/api/health"
echo

echo
echo "Готово. Локально: http://127.0.0.1:${PORT}"
echo "Наружу отдаёт nginx: https://${DOMAIN} — настройка в deploy/setup-nginx.sh"
