"""Сверка весов модели с контрольными суммами.

Постановка требует, чтобы по репозиторию можно было воспроизвести результат.
Веса лежат в репозитории, но «лежат» и «те самые» — разные утверждения: файл
мог быть перезаписан неудачным прогоном обучения, подмениться при слиянии веток
или побиться при переносе. Эта команда отвечает на вопрос одной строкой.

    python -m src.cli.verify_weights

Суммы считаются по SHA-256 и хранятся рядом с весами в models/CHECKSUMS.txt.
Пересчитать их после сознательного переобучения:

    python -m src.cli.verify_weights --update
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

DEFAULT_LIST = Path("models/CHECKSUMS.txt")

#: Файлы, за которые отвечает эта проверка. Порядок фиксированный — так диффы
#: файла сумм читаются глазами, а не воспринимаются как перетасовка.
TRACKED = (
    "models/main/main_model.pkl",
    "models/main/main_model.json",
    "models/baseline.json",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_list(path: Path) -> dict[str, str]:
    known: dict[str, str] = {}
    if not path.exists():
        return known
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            known[parts[1].strip()] = parts[0].strip()
    return known


def write_list(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{digest}  {name}\n" for name, digest in rows)
    path.write_text(
        "# Контрольные суммы весов. Сверка: python -m src.cli.verify_weights\n" + body,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", type=Path, default=DEFAULT_LIST)
    parser.add_argument(
        "--update",
        action="store_true",
        help="пересчитать суммы вместо сверки — только после сознательного переобучения",
    )
    args = parser.parse_args()

    rows: list[tuple[str, str]] = []
    missing: list[str] = []
    for name in TRACKED:
        path = Path(name)
        if not path.exists():
            missing.append(name)
            continue
        rows.append((name, sha256_of(path)))

    if args.update:
        if missing:
            raise SystemExit("нечего пересчитывать, не найдены: " + ", ".join(missing))
        write_list(args.list, rows)
        for name, digest in rows:
            print(f"{digest}  {name}")
        print(f"\nсуммы обновлены: {args.list}")
        return

    known = read_list(args.list)
    if not known:
        raise SystemExit(
            f"файл сумм {args.list} не найден или пуст. "
            "Создайте его: python -m src.cli.verify_weights --update"
        )

    bad: list[str] = []
    for name, digest in rows:
        expected = known.get(name)
        if expected is None:
            print(f"  ?  {name} — суммы для файла нет в списке")
            bad.append(name)
        elif expected != digest:
            print(f"  !  {name} — не совпадает")
            print(f"     ожидалось {expected}")
            print(f"     получено  {digest}")
            bad.append(name)
        else:
            print(f"  OK {name}")
    for name in missing:
        print(f"  !  {name} — файла нет")
        bad.append(name)

    if bad:
        print(
            "\nВеса не те, что записаны. Это не обязательно поломка: так бывает после "
            "переобучения. Если переобучение было сознательным — обновите суммы "
            "командой --update и закоммитьте их вместе с весами."
        )
        sys.exit(1)
    print(f"\nВсе веса совпадают с {args.list}.")


if __name__ == "__main__":
    main()
