from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "core"), str(ROOT / "eval")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from judge_cases import load_cases  # noqa: E402
from kb import load_kb  # noqa: E402
from llm import LLMClient  # noqa: E402
from router import route  # noqa: E402

if __name__ == "__main__":
    wanted = sys.argv[1:]
    cases = load_cases(ROOT / "data", ROOT / "eval" / "cases_own.csv")
    cases = [c for c in cases if (c.id in wanted if wanted else c.source == "own")]
    kb = load_kb(ROOT / "data" / "knowledge-base.zip")
    client = LLMClient()
    lines = [f"# Планы роутера ({client.s.model}), вызовов: {len(cases)}", "",
             "| кейс | ожидали | роутер | docs | queries | дата | claims |", "|---|---|---|---|---|---|---|"]
    for c in cases:
        plan = route(client, kb, c.question, None)
        if plan is None:
            row = [c.id, c.scenario, "НЕТ ПЛАНА", "", "", "", ""]
        else:
            row = [c.id, c.scenario, plan.scenario, ", ".join(plan.docs), "; ".join(plan.queries),
                   plan.event_date.isoformat() if plan.event_date else "", "; ".join(plan.claims)]
        lines.append("| " + " | ".join(x.replace("|", "/") for x in row) + " |")
        print(lines[-1], flush=True)
    out = ROOT / "out" / "eval" / "router_plans.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nвызовов chat: {client.calls['chat']}; сохранено: {out}")
