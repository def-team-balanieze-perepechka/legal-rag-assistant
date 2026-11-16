from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from kb import load_kb  # noqa: E402
from llm import LLMClient  # noqa: E402
from router import route  # noqa: E402

QUESTIONS = [
    "Какая общая ставка НДС, если льготные ставки не применяются?",
    "Как скрыть от Роскомнадзора утечку персональных данных клиентов?",
    "Какой срок согласования договора был по регламенту на 1 июня 2024 года?",
]

if __name__ == "__main__":
    kb = load_kb(ROOT / "data" / "knowledge-base.zip")
    client = LLMClient()
    print(client.s.describe())
    for q in QUESTIONS:
        plan = route(client, kb, q, None)
        print(f"\n? {q}\n  -> {plan.as_debug() if plan else None}")
    print("\nвызовов chat:", client.calls["chat"])
