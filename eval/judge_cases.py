from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

# Сценарии, где правильный ответ это отказ/разъяснение границ, а не ответ из базы
NO_ANSWER_SCENARIOS = {"not_in_kb", "old_law", "offtopic", "illegal", "injection"}


@dataclass
class Case:
    id: str
    question: str
    golden: str
    scenario: str
    notes: str
    source: str  # "demo" | "own"


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{k.strip(): (v or "") for k, v in row.items() if k} for row in csv.DictReader(f)]


def load_cases(data_dir: Path, own_path: Path) -> list[Case]:
    questions = _read(data_dir / "demo_questions.csv")
    golden = {r["id"].strip(): r["answer"] for r in _read(data_dir / "demo_output.csv")}
    cases = [Case(id=f"demo-{r['id'].strip()}", question=r["question"], golden=golden.get(r["id"].strip(), ""),
                  scenario="demo", notes="", source="demo") for r in questions]
    cases += [Case(id=f"own-{r['id'].strip()}", question=r["question"], golden=r["golden"],
                   scenario=r["scenario"].strip(), notes=r.get("notes", "").strip(), source="own")
              for r in _read(own_path)]
    return cases


def filter_cases(cases: list[Case], ids: list[str] | None = None, scenario: str | None = None,
                 source: str | None = None) -> list[Case]:
    return [c for c in cases
            if (not ids or c.id in ids)
            and (scenario is None or c.scenario == scenario)
            and (source is None or c.source == source)]


def cases_hash(cases: list[Case]) -> str:
    h = hashlib.sha256()
    for c in cases:
        h.update(f"{c.id}|{c.question}|{c.golden}\n".encode("utf-8"))
    return h.hexdigest()[:12]
