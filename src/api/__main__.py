"""Запуск панели: ``python -m src.api`` → http://127.0.0.1:8000

Адрес и порт можно переопределить переменными FLOODVALUE_HOST и FLOODVALUE_PORT,
комплект запуска — переменной FLOODVALUE_RUN.
"""

from __future__ import annotations

import uvicorn

from src.api import config


def main() -> None:
    host, port = config.host_port()
    uvicorn.run("src.api.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
