from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from assistant import Assistant, parse_dialog  # noqa: E402
from kb import load_kb  # noqa: E402
from llm import LLMClient, env_int  # noqa: E402


def read_csv(path: Path, col: str) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["id"].strip(): r[col] for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--retrieval", action="store_true", help="только поиск, без вызова LLM")
    ap.add_argument("--out", default=str(ROOT / "out"))
    args = ap.parse_args()

    data = Path(args.data)
    questions = read_csv(data / "demo_questions.csv", "question")
    gold = read_csv(data / "demo_output.csv", "answer")
    ids = args.ids or list(questions)

    client = LLMClient()
    assistant = Assistant(load_kb(data / "knowledge-base.zip"), client)
    t0 = time.time()
    assistant.build_index()
    print(f"индекс: dense={assistant.dense_ready}, {time.time() - t0:.1f}s; {client.s.describe()}", file=sys.stderr)

    def run_one(qid: str) -> dict:
        q = questions[qid]
        if args.retrieval:
            last, hist = parse_dialog(q)
            ctx = " ".join(t["content"] for t in hist)
            hits = assistant.retriever.search(last, top_k=assistant.top_k, context=ctx)
            return {"status": "-", "answer": "", "citations": [], "dropped_citations": [],
                    "debug": {"retrieved": [(h.chunk.chunk_id, h.score, h.sources) for h in hits], "seconds": 0}}
        try:
            return assistant.answer(q)
        except Exception as exc:  # отчёт должен собраться, даже если вопрос упал
            return {"status": "crash", "answer": f"{type(exc).__name__}: {exc}", "citations": [],
                    "dropped_citations": [], "debug": {"retrieved": [], "seconds": 0}}

    t1 = time.time()
    with ThreadPoolExecutor(max_workers=env_int("MAX_WORKERS", 4, 1, 16)) as pool:
        results = dict(zip(ids, pool.map(run_one, ids)))
    total = time.time() - t1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "demo_answers.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        w.writerow(["id", "answer"])
        for qid in ids:
            w.writerow([qid, results[qid]["answer"]])

    words = [len(r["answer"].split()) for r in results.values()]
    dropped = sum(len(r.get("dropped_citations", [])) for r in results.values())
    lines = [
        "# Прогон демо-вопросов", "",
        f"- вопросов: {len(ids)}, время: {total:.1f}s, dense: {assistant.dense_ready}",
        f"- вызовы: {client.calls}",
        f"- отброшено выдуманных chunk_id: {dropped}",
        f"- длина ответа, слов: среднее {sum(words) / max(1, len(words)):.0f}, максимум {max(words, default=0)}",
        f"- статусы: { {s: sum(1 for r in results.values() if r['status'] == s) for s in {r['status'] for r in results.values()}} }",
        "",
    ]
    for qid in ids:
        r = results[qid]
        lines += [
            f"## {qid}. [{r['status']}] {r['debug']['seconds']}s", "",
            "**Вопрос:** " + questions[qid].replace("\n", " / "), "",
            "**Ответ:** " + r["answer"], "",
            "**Эталон:** " + gold.get(qid, "—"), "",
            "**Цитаты:** " + (", ".join(c["chunk_id"] for c in r["citations"]) or "—")
            + (f"; отброшены: {r['dropped_citations']}" if r.get("dropped_citations") else ""), "",
            "**Найдено:** " + ", ".join(f"{cid} {'/'.join(src)}" for cid, _, src in r["debug"]["retrieved"][:12]),
            "",
        ]
    (out / "demo_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:7]))
    print(f"отчёт: {out / 'demo_report.md'}")


if __name__ == "__main__":
    main()
