from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "core"), str(ROOT / "eval")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from assistant import Assistant  # noqa: E402
from judge_cases import load_cases  # noqa: E402
from kb import load_kb  # noqa: E402
from llm import LLMClient  # noqa: E402
from router import route  # noqa: E402

DEFAULT = ["own-2", "own-12", "own-15", "own-16", "own-17", "own-18"]


def where(gold: list[str], ids: list[str]) -> str:
    return ", ".join(f"{g}: {'#' + str(ids.index(g) + 1) if g in ids else 'НЕТ'}" for g in gold) or "(эталон без фрагментов)"


if __name__ == "__main__":
    wanted = sys.argv[1:] or DEFAULT
    cases = [c for c in load_cases(ROOT / "data", ROOT / "eval" / "cases_own.csv") if c.id in wanted]
    a = Assistant(load_kb(ROOT / "data" / "knowledge-base.zip"), LLMClient())
    a.build_index()
    print("dense_ready:", a.dense_ready)
    for c in cases:
        gold = [g.strip() for g in (c.notes or "").split(";") if "#" in g]
        plan = route(a.client, a.kb, c.question, None)
        base = [h.chunk.chunk_id for h in a.retriever.search(c.question, top_k=a.top_k)]
        withp = [h.chunk.chunk_id for h in a.retriever.search(c.question, top_k=a.top_k, plan=plan)]
        print(f"\n{c.id} [{c.scenario}] {c.question}")
        print(f"  план: {plan.scenario if plan else None} docs={plan.docs if plan else None}")
        print(f"  без плана ({len(base)}): {where(gold, base)}")
        print(f"  с планом  ({len(withp)}): {where(gold, withp)}")
        print(f"  лишние документы с планом: {sorted({i.split('#')[0] for i in withp} - {g.split('#')[0] for g in gold})}")
    print("\nвызовов chat:", a.client.calls["chat"], "rerank:", a.client.calls["rerank"])
