"""Точка входа платформы: DATA_DIR/questions.csv + база знаний → submission.csv (id,answer)."""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
# локально модули лежат в core/, в сабмите — рядом с solution.py;
# путь добавляем явно, не полагаясь на то, как запущен интерпретатор
sys.path.insert(0, str(HERE / "core") if (HERE / "core").is_dir() else str(HERE))

try:  # локальная разработка: подхватываем .env; на платформе модуля может не быть
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
except ImportError:
    pass

from assistant import Assistant  # noqa: E402
from generate import FAIL_ANSWER  # noqa: E402
from llm import LLMClient, env, env_int  # noqa: E402

KB_NAMES = ("kb.zip", "knowledge-base.zip", "knowledge_base.zip")
MAX_ANSWER_CHARS = 10_000


def find_file(data_dir: Path, *names: str) -> Path:
    for root in (data_dir, HERE / data_dir, Path.cwd()):
        for name in names:
            if (root / name).is_file():
                return root / name
    raise FileNotFoundError(f"не найден ни один из файлов: {', '.join(names)}")


def load_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"id", "question"} <= set(reader.fieldnames):
            raise ValueError("в questions.csv нужны колонки id и question")
        return [(str(r["id"]).strip(), str(r["question"] or "").strip()) for r in reader]


def write_submission(path: Path, ids: list[str], answers: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        w.writerow(["id", "answer"])
        for qid in ids:
            text = " ".join((answers.get(qid) or FAIL_ANSWER).split())[:MAX_ANSWER_CHARS]
            w.writerow([qid, text or FAIL_ANSWER])


def run(data_dir: Path, out_path: Path, client=None) -> dict[str, str]:
    t0 = time.time()
    budget = env_int("TIME_BUDGET_SEC", 900, 60, 1150)
    workers = env_int("MAX_WORKERS", 4, 1, 16)
    questions = load_questions(find_file(data_dir, "questions.csv"))
    ids = [qid for qid, _ in questions]
    answers: dict[str, str] = {}
    write_submission(out_path, ids, answers)  # непустой файл есть с самого начала

    client = client or LLMClient()
    assistant = Assistant.from_path(find_file(data_dir, *KB_NAMES), client)
    print(f"[solution] docs={len(assistant.kb.documents)} chunks={len(assistant.kb.chunks)} "
          f"questions={len(questions)} workers={workers} top_k={assistant.top_k} budget={budget}s", file=sys.stderr)
    if hasattr(client, "s"):
        print(f"[solution] {client.s.describe()}", file=sys.stderr)
    assistant.build_index()
    print(f"[solution] индекс: dense={assistant.dense_ready} за {time.time() - t0:.1f}s", file=sys.stderr)

    def work(qid: str, question: str) -> tuple[str, str]:
        if time.time() - t0 > budget:
            return qid, FAIL_ANSWER
        try:
            return qid, assistant.answer(question)["answer"]
        except Exception as exc:
            print(f"[solution] вопрос {qid}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return qid, FAIL_ANSWER

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, qid, q) for qid, q in questions]
        for n, fut in enumerate(as_completed(futures), 1):
            qid, text = fut.result()
            answers[qid] = text
            if n % 10 == 0 or n == len(futures):
                print(f"[solution] {n}/{len(futures)} за {time.time() - t0:.0f}s", file=sys.stderr)
                write_submission(out_path, ids, answers)
    write_submission(out_path, ids, answers)
    calls = getattr(client, "calls", {})
    print(f"[solution] готово: {out_path} за {time.time() - t0:.1f}s; вызовы {calls}", file=sys.stderr)
    return answers


def main() -> None:
    run(Path(env("DATA_DIR", "data")), Path(os.getenv("SUBMISSION_PATH", "submission.csv")))


if __name__ == "__main__":
    main()
