from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "eval"))

from judge_cases import Case  # noqa: E402
from judge_checks import BASELINE_FALLBACK  # noqa: E402

ANSWER_FIELDS = ["id", "answer", "status", "raw_ok", "seconds", "citations"]


def _read_answers(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[r["id"]] = {"answer": r["answer"], "status": r["status"],
                            "raw_ok": r["raw_ok"] == "True",
                            "seconds": float(r["seconds"] or 0),
                            "citations": json.loads(r["citations"] or "[]")}
    return out


def answer_ours(cases: list[Case], assistant, path: Path) -> dict[str, dict]:
    """Ответы агента, уже записанные в path не генерируются заново."""
    done = _read_answers(path)
    todo = [c for c in cases if c.id not in done]
    if todo:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.is_file()
        with path.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            if new_file:
                w.writerow(ANSWER_FIELDS)
            for n, c in enumerate(todo, 1):
                t0 = time.time()
                try:
                    res = assistant.answer(c.question)
                except Exception as exc:
                    res = {"answer": f"{type(exc).__name__}: {exc}", "status": "crash", "citations": []}
                row = {"answer": res["answer"], "status": res.get("status", "answer"),
                       "raw_ok": res.get("raw_ok", True), "seconds": round(time.time() - t0, 2),
                       "citations": [{"doc_id": x["doc_id"], "chunk_id": x.get("chunk_id")}
                                     for x in res.get("citations", [])]}
                w.writerow([c.id, row["answer"], row["status"], row["raw_ok"], row["seconds"],
                            json.dumps(row["citations"], ensure_ascii=False)])
                f.flush()
                done[c.id] = row
                print(f"[judge] агент {n}/{len(todo)} {c.id}: {row['status']} {row['seconds']}s", file=sys.stderr)
    return {c.id: done[c.id] for c in cases}


# Настройки нашего агента из .env, которые baseline тоже читает: без них он работает
# со своими значениями по умолчанию, как у организаторов
AGENT_TUNING_ENV = ("TOP_K", "MAX_TOKENS", "RERANKER_TOP_N", "CHUNKER", "MAX_WORKERS", "TEMPERATURE")
BASELINE_BATCH = 10


def baseline_env(environ, data_dir: Path) -> dict[str, str]:
    env = {k: v for k, v in environ.items() if k not in AGENT_TUNING_ENV}
    env.update({"DATA_DIR": str(data_dir), "PYTHONIOENCODING": "utf-8",
                "MAX_TOKENS": environ.get("BASELINE_MAX_TOKENS") or "4096"})
    return env


def run_baseline(questions: list[tuple[str, str]], data_dir: Path) -> dict[str, str]:
    """Стартовый solution.py организаторов во временной папке; {} при любой ошибке."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            with zipfile.ZipFile(data_dir / "sample_submission.zip") as z:
                z.extractall(tmp_path)
            shutil.copy(data_dir / "knowledge-base.zip", tmp_path / "knowledge-base.zip")
            with (tmp_path / "questions.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, quoting=csv.QUOTE_ALL)
                w.writerow(["id", "question"])
                w.writerows(questions)
            proc = subprocess.run([sys.executable, "solution.py"], cwd=tmp_path, timeout=3600,
                                  env=baseline_env(os.environ, tmp_path), capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
            if proc.returncode != 0:
                print(f"[judge] baseline завершился с кодом {proc.returncode}: {proc.stderr[-500:]}", file=sys.stderr)
                return {}
            with (tmp_path / "submission.csv").open(encoding="utf-8-sig", newline="") as f:
                return {r["id"]: r["answer"] for r in csv.DictReader(f)}
        except (OSError, subprocess.SubprocessError, zipfile.BadZipFile, KeyError) as exc:
            print(f"[judge] baseline не запустился: {type(exc).__name__}: {exc}", file=sys.stderr)
            return {}


def _baseline_key(c: Case, model: str) -> str:
    return f"{c.id}|{hashlib.sha1(c.question.encode('utf-8')).hexdigest()[:8]}|{model}"


def answer_baseline(cases: list[Case], cache_path: Path, data_dir: Path, model: str,
                    runner=run_baseline) -> dict[str, str]:
    cache: dict[str, str] = {}
    if cache_path.is_file():
        with cache_path.open(encoding="utf-8", newline="") as f:
            cache = {r["key"]: r["answer"] for r in csv.DictReader(f)}
    todo = [c for c in cases if _baseline_key(c, model) not in cache]
    for i in range(0, len(todo), BASELINE_BATCH):
        batch = todo[i:i + BASELINE_BATCH]
        print(f"[judge] baseline: вопросы {i + 1}-{i + len(batch)} из {len(todo)}", file=sys.stderr)
        got = runner([(c.id, c.question) for c in batch], data_dir)
        # Заглушки не кешируем: в следующий раз попробуем снова
        fresh = {_baseline_key(c, model): got[c.id] for c in batch
                 if (got.get(c.id) or "").strip() not in ("", BASELINE_FALLBACK)}
        if fresh:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not cache_path.is_file()
            with cache_path.open("a", encoding="utf-8", newline="") as f:
                w = csv.writer(f, quoting=csv.QUOTE_ALL)
                if new_file:
                    w.writerow(["key", "answer"])
                w.writerows(fresh.items())
            cache.update(fresh)
    return {c.id: cache.get(_baseline_key(c, model), BASELINE_FALLBACK) for c in cases}


def unload_model(models: list[str], base_url: str) -> None:
    """
    Выгрузить генератор и эмбеддер из видеопамяти перед
    судьей (только локальная Ollama).
    """
    if "11434" not in (base_url or ""):
        return
    for model in filter(None, models):
        try:
            subprocess.run(["ollama", "stop", model], capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            pass


from judge_cases import cases_hash, filter_cases, load_cases  # noqa: E402
from judge_checks import check_answer  # noqa: E402
from judge_core import Judge, aggregate, disagrees, judge_client_from_env, pair_score  # noqa: E402

HISTORY_FIELDS = ["time", "name", "cases_hash", "n", "score", "match_rate", "win_rate", "disagreement",
                  "stubs", "bad_refs", "judge_model", "generator_model"]
METRIC_KEYS = ("score", "match_rate", "win_rate", "disagreement")


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def append_history(path: Path, row: dict) -> dict | None:
    """
    Дописывает строку.
    Возвращает последнюю прежнюю строку на том же наборе кейсов.
    """
    prev = None
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as f:
            same = [r for r in csv.DictReader(f) if r["cases_hash"] == row["cases_hash"]]
        prev = same[-1] if same else None
    new_file = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)
    return prev


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _flag_summary(flag_lists: list[list[str]]) -> str:
    counts: dict[str, int] = {}
    for fl in flag_lists:
        for f in fl:
            counts[f] = counts.get(f, 0) + 1
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "без флагов"


def _by_group(rows: list[dict]) -> dict[str, dict]:
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(f"source={r['source']}", []).append(r["s"])
        groups.setdefault(r["scenario"], []).append(r["s"])
    return {g: aggregate(s) for g, s in sorted(groups.items())}


def write_report(path: Path, rows: list[dict], metrics: dict, prev: dict | None) -> None:
    lines = [f"# Судья: {metrics['name']}", ""]
    if "score" in metrics:
        if prev:
            lines += ["| Метрика | Значение | Δ к прошлому прогону |", "|---|---|---|"]
        else:
            lines += ["| Метрика | Значение |", "|---|---|"]
        for key in METRIC_KEYS:
            cell = f"| {key} | {metrics[key]:.3f} |"
            if prev:
                cell += f" Δ {metrics[key] - float(prev[key]):+.3f} (было {float(prev[key]):.3f}, {prev['name']}) |"
            lines.append(cell)
        lines += ["", "- Score = 0.7·MatchRate + 0.3·WinRate (формула платформы); "
                      f"W/T/L: {metrics['wins']}/{metrics['ties']}/{metrics['losses']}; "
                      f"ошибок судьи (раундов): {metrics['judge_errors']}; кейсов: {metrics['cases']}", "",
                  "## По сценариям", "", "| Группа | n | Score | MatchRate | WinRate |", "|---|---|---|---|---|"]
        for group, m in metrics["by_group"].items():
            lines.append(f"| {group} | {m['n']} | {m['score']:.3f} | {m['match_rate']:.3f} | {m['win_rate']:.3f} |")
        lines.append("")
    lines += ["## Проверки кодом", "",
              f"- наш агент: {metrics['flags_ours']}",
              f"- baseline: {metrics.get('flags_base', 'не запускался')}",
              f"- время на вопрос (агент): p50 {metrics['p50']:.1f}s, p95 {metrics['p95']:.1f}s", "",
              "## По вопросам (сначала худшие)", ""]
    for r in sorted(rows, key=lambda r: (r["s"] is None, r["s"] if r["s"] is not None else 0.0)):
        lines += [f"### {r['id']} [{r['scenario']}] s={_fmt(r['s'])} {' '.join(r['flags'])}".rstrip(), "",
                  "**Вопрос:** " + r["question"].replace("\n", " / "), "",
                  "**Эталон:** " + r["golden"], "",
                  f"**Наш ({r['status']}):** " + r["ours"], ""]
        if "base" in r:
            lines += [f"**Baseline:** {r['base']} {' '.join(r['base_flags'])}".rstrip(), ""]
        if "base" in r and "stub" in r["flags"]:
            lines.append("- заглушка: s=0 без судьи")
        for i, rd in enumerate(r.get("rounds", []), 1):
            fab = " · выдумка в нашем" if rd.ours_fabricated else ""
            lines.append(f"- раунд {i}: {rd.verdict or 'ошибка судьи'}{fab} — {rd.reasoning}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _count(flag_lists: list[list[str]], flag: str) -> int:
    return sum(flag in f for f in flag_lists)


def run(name: str, data_dir: Path, out_root: Path, *, own_path: Path | None = None, ids=None, scenario=None,
        source=None, checks_only: bool = False, assistant=None, judge=None, baseline_runner=run_baseline) -> dict:
    from assistant import Assistant
    from kb import load_kb
    from llm import LLMClient

    cases = filter_cases(load_cases(data_dir, own_path or ROOT / "eval" / "cases_own.csv"), ids, scenario, source)
    run_dir = out_root / name
    if assistant is None:
        assistant = Assistant(load_kb(data_dir / "knowledge-base.zip"), LLMClient())
        if any(c.id not in _read_answers(run_dir / "answers.csv") for c in cases):
            assistant.build_index()
            if not assistant.dense_ready and os.getenv("ALLOW_BM25") != "1":
                raise SystemExit("Эмбеддинги недоступны (запущена ли Ollama с bge-m3?): ответы на одном BM25 "
                                 "несравнимы с прежними прогонами. Запустите Ollama или задайте ALLOW_BM25=1.")
    kb = assistant.kb
    ours = answer_ours(cases, assistant, run_dir / "answers.csv")
    gen_settings = getattr(assistant.client, "s", None)
    gen_model = getattr(gen_settings, "model", "fake")

    rows = []
    for c in cases:
        o = ours[c.id]
        rows.append({"id": c.id, "scenario": c.scenario, "source": c.source, "question": c.question,
                     "golden": c.golden, "ours": o["answer"], "status": o["status"], "s": None,
                     "flags": check_answer(o["answer"], kb, c, o["status"], o["citations"], o["raw_ok"])})
    secs = [ours[c.id]["seconds"] for c in cases]
    flags = [r["flags"] for r in rows]
    metrics = {"name": name, "cases": len(cases), "flags_ours": _flag_summary(flags),
               "stubs": _count(flags, "stub"), "bad_refs": _count(flags, "bad_ref"),
               "p50": _pct(secs, 0.5), "p95": _pct(secs, 0.95)}
    if checks_only:
        write_report(run_dir / "report.md", rows, metrics, None)
        return metrics

    base = answer_baseline(cases, out_root / "baseline" / "answers.csv", data_dir, gen_model, runner=baseline_runner)
    if judge is None:
        unload_model([gen_model, getattr(gen_settings, "emb_model", "")], getattr(gen_settings, "base_url", ""))
        client = judge_client_from_env()
        judge = Judge(client, out_root / "judge_cache.jsonl", client.s.model)
    scores, dis, errors, both_ok = [], 0, 0, 0
    for n, (c, r) in enumerate(zip(cases, rows), 1):
        r["base"] = base[c.id]
        r["base_flags"] = check_answer(base[c.id], kb, c)
        if "stub" in r["flags"]:
            # Заглушка не может выиграть, а локальный судья путал стороны и ставил ей 1.0
            r["rounds"], r["s"] = [], 0.0
            scores.append(0.0)
            print(f"[judge] судья {n}/{len(cases)} {c.id}: заглушка, s=0", file=sys.stderr)
            continue
        r1, r2 = judge.judge_pair(c, r["ours"], r["base"])
        r["rounds"], r["s"] = [r1, r2], pair_score(r1, r2)
        scores.append(r["s"])
        dis += disagrees(r1, r2)
        errors += r1.error + r2.error
        both_ok += not (r1.error or r2.error)
        print(f"[judge] судья {n}/{len(cases)} {c.id}: s={r['s']}", file=sys.stderr)
    metrics.update(aggregate(scores))
    metrics.update({"disagreement": dis / both_ok if both_ok else 0.0, "judge_errors": errors,
                    "flags_base": _flag_summary([r["base_flags"] for r in rows]), "by_group": _by_group(rows)})
    prev = append_history(out_root / "history.csv", {
        "time": time.strftime("%Y-%m-%d %H:%M"), "name": name, "cases_hash": cases_hash(cases),
        **{k: round(metrics[k], 4) for k in METRIC_KEYS}, "n": metrics["n"],
        "stubs": metrics["stubs"], "bad_refs": metrics["bad_refs"], "judge_model": judge.model,
        "generator_model": gen_model})
    write_report(run_dir / "report.md", rows, metrics, prev)
    return metrics


def main() -> None:
    import argparse

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description="Локальный судья: агент против baseline по методике платформы")
    ap.add_argument("--name", required=True, help="имя версии агента (папка out/eval/<name>)")
    ap.add_argument("--checks-only", action="store_true", help="только ответы агента и проверки кодом")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--scenario")
    ap.add_argument("--source", choices=["demo", "own"])
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "out" / "eval"))
    args = ap.parse_args()
    m = run(args.name, Path(args.data), Path(args.out), ids=args.ids, scenario=args.scenario,
            source=args.source, checks_only=args.checks_only)
    keys = (*METRIC_KEYS, "judge_errors", "stubs", "bad_refs")
    print(" ".join(f"{k}={_fmt(m[k])}" for k in keys if k in m))
    print(f"отчёт: {Path(args.out) / args.name / 'report.md'}")


if __name__ == "__main__":
    main()
