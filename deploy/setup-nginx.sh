#!/usr/bin/env bash
# Настройка обратного прокси и сертификата для «Водополя».
#
# Запускать после deploy/deploy.sh, из корня репозитория:
#   bash deploy/setup-nginx.sh
#
# Конфигурация здесь дублирует deploy/nginx-vodopol.conf намеренно: скрипт должен
# работать, даже если файл конфигурации не доехал на сервер. Правите одно —
# правьте и второе.
set -e

PORT=8020
DOMAIN=vodopol.clv-digital.tech

echo "== контейнер"
curl -fsS "http://127.0.0.1:${PORT}/api/health" && echo " <- жив"

echo "== кладу конфигурацию nginx"
sudo tee /etc/nginx/sites-available/vodopol > /dev/null <<'CONF'
server {
    listen 80;
    listen [::]:80;
    server_name vodopol.clv-digital.tech;

    proxy_connect_timeout 15s;
    proxy_send_timeout    120s;
    proxy_read_timeout    120s;

    gzip on;
    gzip_types application/json application/geo+json application/javascript text/css text/plain image/svg+xml;
    gzip_min_length 1024;
    client_max_body_size 1m;

    location / {
        proxy_pass http://127.0.0.1:8020;
        proxy_http_version 1.1;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
    }
}
CONF

sudo ln -sf /etc/nginx/sites-available/vodopol /etc/nginx/sites-enabled/vodopol
sudo nginx -t 2>&1 | tail -2
sudo systemctl reload nginx && echo "nginx перезагружен"

echo "== проверка через nginx"
curl -s -o /dev/null -w "по http: %{http_code}\n" -H "Host: ${DOMAIN}" http://127.0.0.1/api/health

echo "== проверяю, что заголовок границ доходит до клиента"
# Если X-Bounds потеряется на прокси, карта откроется пустой, а в консоли браузера
# не будет ни одной ошибки — поэтому проверяем здесь, а не на защите.
if curl -s -D - -o /dev/null -H "Host: ${DOMAIN}" \
       http://127.0.0.1/api/raster/probability.png | grep -qi "^x-bounds:"; then
    echo "X-Bounds на месте"
else
    echo "X-Bounds не дошёл — карта будет пустой, проверьте фильтры заголовков" >&2
fi

echo "== выпускаю сертификат"
# Почта уходит в Let's Encrypt и нужна им для писем об истечении сертификата.
# Задайте её переменной окружения, иначе certbot спросит в диалоге:
#   CERTBOT_EMAIL=you@example.com bash deploy/setup-nginx.sh
if [ -n "${CERTBOT_EMAIL:-}" ]; then
    sudo certbot --nginx -d "${DOMAIN}" --non-interactive --agree-tos \
         -m "${CERTBOT_EMAIL}" --redirect || \
         echo "certbot не справился — выпустите сертификат вручную: sudo certbot --nginx -d ${DOMAIN}"
else
    sudo certbot --nginx -d "${DOMAIN}" --redirect || \
         echo "certbot не справился — выпустите сертификат вручную: sudo certbot --nginx -d ${DOMAIN}"
fi

echo
echo "Готово: https://${DOMAIN}"
