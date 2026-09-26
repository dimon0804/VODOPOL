#!/usr/bin/env bash
# Полная установка «Водополя» на чистый сервер одной командой.
#
#   ssh root@<адрес>
#   curl -fsSL https://gitverse.ru/hackrus.experts/kosmo-moskva_clvdigital_8/raw/branch/master/deploy/server-setup.sh | bash
#
# Или, если репозиторий уже склонирован:
#
#   bash deploy/server-setup.sh
#
# Делает всё подряд и ничего не спрашивает: ставит Docker, забирает код, собирает
# образ, поднимает контейнер, настраивает nginx и выпускает сертификат.
# Идемпотентен — повторный запуск обновляет код и пересобирает, не ломая уже
# работающее. Ошибка на выпуске сертификата не валит установку: панель к этому
# моменту уже работает по HTTP, и сертификат можно выпустить отдельно.
set -euo pipefail

REPO="${REPO:-https://gitverse.ru/hackrus.experts/kosmo-moskva_clvdigital_8.git}"
DIR="${DIR:-/opt/vodopol}"
DOMAIN="${DOMAIN:-vodopol.clv-digital.tech}"
PORT="${PORT:-8020}"
EMAIL="${EMAIL:-admin@clv-digital.tech}"
NAME=vodopol
IMAGE=vodopol:prod

say() { printf '\n==> %s\n' "$*"; }

# ── 1. Системные пакеты ──────────────────────────────────────────────────────
say "Системные пакеты"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    ca-certificates curl git nginx ufw >/dev/null

# ── 2. Docker ────────────────────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
    say "Ставлю Docker"
    curl -fsSL https://get.docker.com | sh >/dev/null
else
    say "Docker уже стоит: $(docker --version)"
fi
systemctl enable --now docker >/dev/null 2>&1 || true

# ── 3. Код ───────────────────────────────────────────────────────────────────
if [ -d "$DIR/.git" ]; then
    say "Обновляю код в $DIR"
    git -C "$DIR" fetch --all --quiet
    git -C "$DIR" reset --hard origin/master --quiet
else
    say "Забираю код в $DIR"
    rm -rf "$DIR"
    git clone --quiet "$REPO" "$DIR"
fi
cd "$DIR"
echo "    ревизия $(git rev-parse --short HEAD)"

# Комплекты запусков лежат в репозитории, поэтому показывать есть что сразу.
COUNT=$(ls -d outputs/*/ 2>/dev/null | wc -l || echo 0)
echo "    комплектов запусков: $COUNT"
if [ "$COUNT" -eq 0 ]; then
    echo "    ВНИМАНИЕ: комплектов нет, панель откроется в демо-режиме" >&2
fi

# ── 4. Образ и контейнер ─────────────────────────────────────────────────────
say "Собираю образ"
docker build -q -t "$IMAGE" . >/dev/null

say "Перезапускаю контейнер"
docker rm -f "$NAME" >/dev/null 2>&1 || true
# Слушаем только локальный адрес: наружу отдаёт nginx, у приложения нет забот
# про TLS. Комплекты и веса монтируются из каталога репозитория, поэтому
# пересоздание контейнера ничего не теряет.
docker run -d --name "$NAME" --restart unless-stopped \
    -p 127.0.0.1:"$PORT":8000 \
    -v "$DIR/outputs":/app/outputs \
    -v "$DIR/models":/app/models:ro \
    -v "$DIR/data/cache":/app/data/cache \
    -e FLOODVALUE_RUN="${FLOODVALUE_RUN:-}" \
    "$IMAGE" >/dev/null

say "Жду, пока сервис ответит"
for i in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
        echo "    живой на $i-й секунде"
        break
    fi
    sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null || {
    echo "Сервис не поднялся. Логи:" >&2
    docker logs --tail 40 "$NAME" >&2
    exit 1
}

# ── 5. nginx ─────────────────────────────────────────────────────────────────
say "Настраиваю nginx на $DOMAIN"
cat >/etc/nginx/sites-available/vodopol <<NGINX
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;

    # Растровые слои карты приходят с границами в заголовке X-Bounds. Без него
    # карта открывается пустой и ни одной ошибки в консоли браузера при этом нет,
    # поэтому заголовок пробрасывается явно и проверяется ниже.
    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_pass_header X-Bounds;
        proxy_read_timeout 300s;
        client_max_body_size 32m;
    }
}
NGINX
ln -sf /etc/nginx/sites-available/vodopol /etc/nginx/sites-enabled/vodopol
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx

# ── 6. Брандмауэр ────────────────────────────────────────────────────────────
say "Открываю порты"
ufw allow OpenSSH >/dev/null 2>&1 || true
ufw allow 'Nginx Full' >/dev/null 2>&1 || true
ufw --force enable >/dev/null 2>&1 || true

# ── 7. Сертификат ────────────────────────────────────────────────────────────
say "Выпускаю сертификат"
if ! command -v certbot >/dev/null 2>&1; then
    apt-get install -y -qq --no-install-recommends certbot python3-certbot-nginx >/dev/null
fi
if certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect; then
    echo "    сертификат выпущен, включено перенаправление на HTTPS"
else
    echo "    сертификат выпустить не удалось — панель работает по HTTP" >&2
    echo "    повторить: certbot --nginx -d $DOMAIN" >&2
fi

# ── 8. Проверка снаружи ──────────────────────────────────────────────────────
say "Проверяю снаружи"
for url in "https://$DOMAIN/api/health" "http://$DOMAIN/api/health"; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 "$url" || echo 000)
    printf '    %-48s %s\n' "$url" "$code"
    [ "$code" = "200" ] && OK="$url"
done

# Заголовок X-Bounds доходит через прокси? Без него карта пустая.
BOUNDS=$(curl -sI -m 20 "https://$DOMAIN/api/raster/probability.png" 2>/dev/null | grep -ci '^x-bounds' || echo 0)
echo "    заголовок X-Bounds через прокси: $([ "$BOUNDS" -gt 0 ] && echo есть || echo НЕТ)"

say "Готово"
echo "    Панель: https://$DOMAIN"
echo "    Логи:   docker logs -f $NAME"
echo "    Обновить: bash $DIR/deploy/server-setup.sh"
