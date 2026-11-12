from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "eval"))

from judge_cases import Case, cases_hash, filter_cases, load_cases  # noqa: E402

DATA = ROOT / "data"
KB_PATH = DATA / "knowledge-base.zip"
needs_data = pytest.mark.skipif(not KB_PATH.exists(), reason="нет data/knowledge-base.zip")


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(header)
        w.writerows(rows)


def _case(cid="own-1", scenario="fact", question="Вопрос?", golden="Эталон.", source="own") -> Case:
    return Case(id=cid, question=question, golden=golden, scenario=scenario, notes="", source=source)



DIALOG = 'Пользователь: а "кавычки"\nАссистент: б\nПользователь: в?'


def _demo_and_own(tmp_path: Path) -> Path:
    _write_csv(tmp_path / "demo_questions.csv", ["id", "question"], [["1", DIALOG]])
    _write_csv(tmp_path / "demo_output.csv", ["id", "answer"], [["1", "Эталон демо (п. 1.1)."]])
    own = tmp_path / "cases_own.csv"
    _write_csv(own, ["id", "question", "golden", "scenario", "notes"],
               [["1", "Свой вопрос?", "Свой эталон.", "fact", "contract_a-08-2024#p3.1"]])
    return own


def test_load_cases_merges_demo_and_own(tmp_path):
    own = _demo_and_own(tmp_path)
    cases = load_cases(tmp_path, own)
    assert [c.id for c in cases] == ["demo-1", "own-1"]
    assert cases[0].scenario == "demo" and cases[0].source == "demo"
    assert cases[0].golden == "Эталон демо (п. 1.1)."
    assert "\n" in cases[0].question and '"' in cases[0].question
    assert cases[1].scenario == "fact" and cases[1].source == "own"
    assert cases[1].notes == "contract_a-08-2024#p3.1"


def test_load_cases_without_own_file(tmp_path):
    _demo_and_own(tmp_path)
    cases = load_cases(tmp_path, tmp_path / "missing.csv")
    assert [c.id for c in cases] == ["demo-1"]


def test_filter_cases_by_ids_scenario_source(tmp_path):
    cs = load_cases(tmp_path, _demo_and_own(tmp_path))
    assert [c.id for c in filter_cases(cs, scenario="fact")] == ["own-1"]
    assert [c.id for c in filter_cases(cs, source="demo")] == ["demo-1"]
    assert [c.id for c in filter_cases(cs, ids=["own-1"])] == ["own-1"]
    assert [c.id for c in filter_cases(cs)] == ["demo-1", "own-1"]


def test_cases_hash_changes_with_golden():
    a = [_case(golden="один")]
    b = [_case(golden="два")]
    assert cases_hash(a) != cases_hash(b)
    assert len(cases_hash(a)) == 12
    assert cases_hash(a) == cases_hash([_case(golden="один")])


from judge_core import Round, aggregate, disagrees, pair_score, round_score  # noqa: E402


def test_round_score_maps_labels_from_our_side():
    assert round_score("B_much_better", "B") == 1.0 and round_score("B_much_better", "A") == 0.0
    assert round_score("A_better", "A") == 0.75 and round_score("A_better", "B") == 0.25
    assert round_score("equal", "B") == 0.5


def test_pair_score_averages_and_zeroes_fabrication():
    assert pair_score(Round("B_better"), Round("A_much_better")) == 0.875
    assert pair_score(Round("B_better", ours_fabricated=True), Round("A_much_better")) == 0.0
    assert pair_score(Round(None, error=True), Round("equal")) == 0.5
    assert pair_score(Round(None, error=True), Round(None, error=True)) is None


def test_disagrees():
    assert disagrees(Round("B_better"), Round("B_better"))
    assert not disagrees(Round("B_better"), Round("A_better"))
    assert not disagrees(Round("equal"), Round("B_better"))
    assert not disagrees(Round(None, error=True), Round("B_better"))


def test_aggregate_matches_platform_formula():
    m = aggregate([1.0, 0.5, 0.25, None])
    assert m["n"] == 3 and m["wins"] == 1 and m["ties"] == 1 and m["losses"] == 1
    assert m["match_rate"] == pytest.approx(0.58333, abs=1e-4)
    assert m["win_rate"] == 0.5
    assert m["score"] == pytest.approx(0.7 * 0.58333 + 0.3 * 0.5, abs=1e-4)
    assert aggregate([0.5, 0.5])["win_rate"] == 0.5
    assert aggregate([])["n"] == 0



from judge_core import PROMPT_VERSION, Judge, build_messages, judge_client_from_env, parse_round  # noqa: E402


class FakeJudgeClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0
        self.messages = []

    def chat(self, messages, max_tokens=None):
        self.calls += 1
        self.messages.append(messages)
        return self.reply(messages) if callable(self.reply) else self.reply


def _verdict(v, a_fab=False, b_fab=False):
    return json.dumps({"a_fabricated": a_fab, "b_fabricated": b_fab, "a_unjustified_refusal": False,
                       "b_unjustified_refusal": False, "reasoning": "x", "verdict": v})


def _a_text(messages) -> str:
    return messages[-1]["content"].split("ОТВЕТ A:")[1].split("ОТВЕТ B:")[0]


def test_build_messages_has_question_golden_and_both_answers():
    msgs = build_messages(_case(question="Q?", golden="G."), "ответ один", "ответ два")
    user = msgs[-1]["content"]
    assert msgs[0]["role"] == "system" and "JSON" in msgs[0]["content"]
    assert "Q?" in user and "G." in user
    assert user.index("ОТВЕТ A:") < user.index("ответ один") < user.index("ОТВЕТ B:") < user.index("ответ два")


def test_parse_round_reads_our_side():
    raw = _verdict("B_better", a_fab=True)
    assert parse_round(raw, "A").ours_fabricated is True
    r = parse_round(raw, "B")
    assert r.ours_fabricated is False and r.verdict == "B_better" and not r.error


def test_parse_round_errors():
    assert parse_round("не json", "B").error
    assert parse_round('{"verdict": "лучше A"}', "B").error


def test_judge_pair_swaps_sides(tmp_path):
    client = FakeJudgeClient(lambda m: _verdict("A_better" if "НАШ" in _a_text(m) else "B_better"))
    judge = Judge(client, tmp_path / "cache.jsonl", "m")
    r1, r2 = judge.judge_pair(_case(), ours="НАШ ответ", baseline="базовый")
    assert pair_score(r1, r2) == 0.75
    assert "базовый" in _a_text(client.messages[0]) and "НАШ ответ" in _a_text(client.messages[1])


def test_judge_retries_once_then_marks_error(tmp_path):
    client = FakeJudgeClient("мусор")
    r1, r2 = Judge(client, tmp_path / "cache.jsonl", "m").judge_pair(_case(), "x", "y")
    assert client.calls == 4 and r1.error and r2.error
    assert pair_score(r1, r2) is None


def test_judge_cache_hit_and_invalidation(tmp_path):
    cache = tmp_path / "cache.jsonl"
    c1, c2, c3 = (FakeJudgeClient(_verdict("equal")) for _ in range(3))
    Judge(c1, cache, "m").judge_pair(_case(), "x", "y")
    assert c1.calls == 2
    r1, r2 = Judge(c2, cache, "m").judge_pair(_case(), "x", "y")
    assert c2.calls == 0 and r1.verdict == "equal"
    Judge(c3, cache, "m").judge_pair(_case(), "x2", "y")
    assert c3.calls == 2


def test_judge_cache_depends_on_prompt_version(tmp_path, monkeypatch):
    import judge_core

    cache = tmp_path / "cache.jsonl"
    Judge(FakeJudgeClient(_verdict("equal")), cache, "m").judge_pair(_case(), "x", "y")
    monkeypatch.setattr(judge_core, "PROMPT_VERSION", PROMPT_VERSION + "-next")
    c = FakeJudgeClient(_verdict("equal"))
    Judge(c, cache, "m").judge_pair(_case(), "x", "y")
    assert c.calls == 2


def test_judge_errors_are_not_cached(tmp_path):
    cache = tmp_path / "cache.jsonl"
    Judge(FakeJudgeClient("мусор"), cache, "m").judge_pair(_case(), "x", "y")
    c = FakeJudgeClient(_verdict("equal"))
    Judge(c, cache, "m").judge_pair(_case(), "x", "y")
    assert c.calls == 2


def test_judge_client_from_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://gen/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen3:4b")
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("JUDGE_MAX_TOKENS", raising=False)
    c = judge_client_from_env()
    assert c.s.model == "qwen3:8b" and c.s.base_url == "http://gen/v1" and c.s.max_tokens == 8192
    assert c.s.temperature == 0.0
    monkeypatch.setenv("JUDGE_MODEL", "big")
    assert judge_client_from_env().s.model == "big"



import generate as gen  # noqa: E402
from judge_checks import BASELINE_FALLBACK, check_answer  # noqa: E402


@pytest.fixture(scope="module")
def kb():
    if not KB_PATH.exists():
        pytest.skip("нет data/knowledge-base.zip")
    import kb as kb_module
    return kb_module.load_kb(KB_PATH)


@needs_data
def test_stub_flags(kb):
    assert "stub" in check_answer(gen.FAIL_ANSWER, kb, _case(), status="error")
    assert "stub" in check_answer(BASELINE_FALLBACK, kb, _case())
    assert "stub" in check_answer("", kb, _case())


@needs_data
def test_raw_text_flag(kb):
    assert "raw_text" in check_answer("Ответ.", kb, _case(), status="answer", citations=[], raw_ok=False)
    assert "raw_text" not in check_answer("Ответ.", kb, _case())


@needs_data
def test_bad_ref_on_missing_clause(kb):
    assert "bad_ref" in check_answer("По договору аренды № А-08/2024 плата 450 000 рублей (п. 9.9).", kb, _case())
    assert "bad_ref" not in check_answer("По договору аренды № А-08/2024 плата 450 000 рублей (п. 3.1).", kb, _case())


@needs_data
def test_bad_ref_accepts_clause_of_other_named_doc(kb):
    text = "По договору аренды № А-08/2024 сумма 12 млн, по регламенту согласования (п. 2.4) нужно одобрение."
    assert "bad_ref" not in check_answer(text, kb, _case(), citations=[{"doc_id": "reglament_soglasovaniya_2023"}])


@needs_data
def test_bad_ref_on_missing_article(kb):
    assert "bad_ref" in check_answer("По ТК РФ, ст. 999, испытание до трёх месяцев.", kb, _case())
    assert "bad_ref" not in check_answer("По ТК РФ, ст. 70, ч. 5, испытание до шести месяцев.", kb, _case())
    assert "bad_ref" not in check_answer("Оператор вправе продолжить обработку (п. 2 ст. 9 152-ФЗ).", kb, _case())
    # честное «статьи нет в базе» — не выдумка
    assert "bad_ref" not in check_answer("Ст. 6 152-ФЗ в базе нет, поэтому основания проверить нельзя.", kb, _case())


@needs_data
def test_foreign_doc(kb):
    cites = [{"doc_id": "reglament_soglasovaniya_2023"}]
    text = "Да, по договору аренды № А-08/2024 (п. 2.4 регламента) нужно одобрение."
    assert "foreign_doc" in check_answer(text, kb, _case(), citations=cites)
    assert "foreign_doc" not in check_answer(text, kb, _case())  # baseline: цитат нет — не проверяем
    addendum = [{"doc_id": "addendum_1_p-17-2024"}]
    assert "foreign_doc" not in check_answer("По договору № П-17/2024 с учётом допсоглашения…", kb, _case(),
                                             citations=addendum)


@needs_data
def test_false_refusal_depends_on_scenario(kb):
    assert "false_refusal" in check_answer("Нет в базе.", kb, _case(scenario="fact"), status="insufficient")
    assert "false_refusal" not in check_answer("Нет в базе.", kb, _case(scenario="not_in_kb"), status="insufficient")
    assert "false_refusal" not in check_answer("Нет в базе.", kb, _case(scenario="fact"))


@needs_data
def test_length_flags(kb):
    assert "too_short" in check_answer("Да.", kb, _case())
    assert "too_long" in check_answer("слово " * 151, kb, _case())
    assert not {"too_short", "too_long"} & set(check_answer("слово " * 40, kb, _case()))


from judge import answer_baseline, answer_ours  # noqa: E402
from test_agent import FakeClient, fragment_ids  # noqa: E402


def _json_reply(messages):
    return json.dumps({"status": "answer", "answer": "Плата — 450 000 рублей в месяц (п. 3.1).",
                       "citations": fragment_ids(messages)[:1]}, ensure_ascii=False)


TWO = [_case("demo-1", question="Какая арендная плата по договору с ООО «Северный Склад»?", source="demo"),
       _case("own-1", question="Какой срок испытания у главного бухгалтера?")]


@needs_data
def test_answer_ours_caches_by_file(tmp_path, kb):
    from assistant import Assistant

    a = Assistant(kb, FakeClient(_json_reply))
    path = tmp_path / "answers.csv"
    r1 = answer_ours(TWO, a, path)
    assert a.client.calls["chat"] == 2
    r2 = answer_ours(TWO, a, path)
    assert a.client.calls["chat"] == 2
    assert r2["own-1"]["status"] == "answer" and r2["own-1"]["answer"] == r1["own-1"]["answer"]
    assert isinstance(r2["demo-1"]["citations"], list) and r2["demo-1"]["raw_ok"] is True


def test_answer_baseline_fills_missing_with_fallback(tmp_path):
    res = answer_baseline(TWO, tmp_path / "base.csv", tmp_path, "m", runner=lambda qs, d: {qs[0][0]: "ответ"})
    assert res["demo-1"] == "ответ" and res["own-1"] == BASELINE_FALLBACK


def test_answer_baseline_uses_cache(tmp_path):
    calls = []

    def runner(qs, d):
        calls.append(qs)
        return {q[0]: "x" for q in qs}

    cache = tmp_path / "base.csv"
    answer_baseline(TWO, cache, tmp_path, "m", runner=runner)
    res = answer_baseline(TWO, cache, tmp_path, "m", runner=runner)
    assert len(calls) == 1 and res == {"demo-1": "x", "own-1": "x"}
    answer_baseline(TWO, cache, tmp_path, "other-model", runner=runner)
    assert len(calls) == 2  # другая модель генератора — другой ключ кеша


def test_answer_baseline_fallback_is_not_cached(tmp_path):
    calls = []

    def runner(qs, d):
        calls.append(qs)
        return {}

    cache = tmp_path / "base.csv"
    answer_baseline(TWO, cache, tmp_path, "m", runner=runner)
    answer_baseline(TWO, cache, tmp_path, "m", runner=runner)
    assert len(calls) == 2



from judge import run  # noqa: E402


def _data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    _write_csv(d / "demo_questions.csv", ["id", "question"],
               [["1", "Какая арендная плата по договору с ООО «Северный Склад»?"],
                ["2", "Какой срок испытания у главного бухгалтера?"]])
    _write_csv(d / "demo_output.csv", ["id", "answer"],
               [["1", "450 000 рублей в месяц (п. 3.1)."], ["2", "Не более шести месяцев (ТК РФ, ст. 70)."]])
    (d / "knowledge-base.zip").write_bytes(KB_PATH.read_bytes())
    return d


def _fakes(kb, out: Path):
    from assistant import Assistant

    a = Assistant(kb, FakeClient(_json_reply))
    fj = FakeJudgeClient(lambda m: _verdict("A_better" if "450 000" in _a_text(m) else "B_better"))
    return a, fj, Judge(fj, out / "judge_cache.jsonl", "m")


def _base(qs, d):
    return {q[0]: "базовый ответ" for q in qs}


@needs_data
def test_run_end_to_end_with_fakes(tmp_path, kb):
    data, out = _data_dir(tmp_path), tmp_path / "out"
    a, fj, judge = _fakes(kb, out)
    m = run("t1", data, out, own_path=tmp_path / "none.csv", assistant=a, judge=judge, baseline_runner=_base)
    report = (out / "t1" / "report.md").read_text(encoding="utf-8")
    hist = list(csv.DictReader((out / "history.csv").open(encoding="utf-8")))
    assert m["n"] == 2 and 0.0 <= m["score"] <= 1.0
    # наш ответ с «450 000» «лучше» в обоих раундах: MatchRate 0.75, WinRate 1 → 0.7·0.75 + 0.3·1
    assert m["score"] == pytest.approx(0.825)
    assert len(hist) == 1 and hist[0]["name"] == "t1" and float(hist[0]["score"]) == pytest.approx(0.825)
    assert "demo-1" in report and "базовый ответ" in report and "Score" in report


@needs_data
def test_rerun_same_name_makes_no_llm_calls(tmp_path, kb):
    data, out = _data_dir(tmp_path), tmp_path / "out"
    a, fj, judge = _fakes(kb, out)
    base_calls = []

    def base(qs, d):
        base_calls.append(qs)
        return _base(qs, d)

    run("t1", data, out, own_path=tmp_path / "none.csv", assistant=a, judge=judge, baseline_runner=base)
    before = (a.client.calls["chat"], fj.calls, len(base_calls))
    run("t1", data, out, own_path=tmp_path / "none.csv", assistant=a,
        judge=Judge(fj, out / "judge_cache.jsonl", "m"), baseline_runner=base)
    assert (a.client.calls["chat"], fj.calls, len(base_calls)) == before


@needs_data
def test_checks_only_skips_baseline_and_judge(tmp_path, kb):
    data, out = _data_dir(tmp_path), tmp_path / "out"
    a, fj, judge = _fakes(kb, out)

    def boom(qs, d):
        raise AssertionError("baseline не должен запускаться")

    m = run("t2", data, out, own_path=tmp_path / "none.csv", checks_only=True, assistant=a, judge=judge,
            baseline_runner=boom)
    assert "score" not in m and m["stubs"] == 0 and fj.calls == 0
    assert (out / "t2" / "report.md").exists()


@needs_data
def test_report_shows_delta_vs_previous_run(tmp_path, kb):
    data, out = _data_dir(tmp_path), tmp_path / "out"
    a, fj, judge = _fakes(kb, out)
    run("t1", data, out, own_path=tmp_path / "none.csv", assistant=a, judge=judge, baseline_runner=_base)
    run("t2", data, out, own_path=tmp_path / "none.csv", assistant=a, judge=judge, baseline_runner=_base)
    assert "Δ" in (out / "t2" / "report.md").read_text(encoding="utf-8")
    assert "Δ" not in (out / "t1" / "report.md").read_text(encoding="utf-8")


from judge_cases import NO_ANSWER_SCENARIOS  # noqa: E402

ALLOWED_SCENARIOS = {"fact", "false_premise", "date_edition", "old_law", "similar_names", "multi_doc",
                     "not_in_kb", "illegal", "injection", "pressure", "offtopic"}


@needs_data
def test_own_cases_are_valid(kb):
    cases = [c for c in load_cases(DATA, ROOT / "eval" / "cases_own.csv") if c.source == "own"]
    assert len(cases) >= 20 and len({c.id for c in cases}) == len(cases)
    for c in cases:
        assert c.scenario in ALLOWED_SCENARIOS, c.id
        assert c.question.strip() and c.golden.strip(), c.id
        assert 10 <= len(c.golden.split()) <= 100, c.id
        for cid in filter(None, (x.strip() for x in c.notes.split(";"))):
            kb.get_chunk(cid)  # KeyError, если фрагмента нет
        if c.scenario not in NO_ANSWER_SCENARIOS - {"illegal"}:
            assert c.notes.strip(), c.id




def test_review_baseline_own_fallback_is_not_cached(tmp_path):
    calls = []

    def runner(qs, d):
        calls.append(qs)
        return {q[0]: BASELINE_FALLBACK for q in qs}  # baseline сам вернул свою заглушку

    cache = tmp_path / "base.csv"
    answer_baseline(TWO, cache, tmp_path, "m", runner=runner)
    answer_baseline(TWO, cache, tmp_path, "m", runner=runner)
    assert len(calls) == 2


def test_review_baseline_env_drops_agent_tuning():
    from judge import baseline_env

    env = baseline_env({"TOP_K": "8", "MAX_TOKENS": "2048", "RERANKER_TOP_N": "5", "CHUNKER": "x",
                        "MAX_WORKERS": "1", "TEMPERATURE": "0.3", "LLM_BASE_URL": "http://o/v1",
                        "LLM_MODEL": "qwen3:4b"}, Path("/tmp/x"))
    assert not {"TOP_K", "RERANKER_TOP_N", "CHUNKER", "MAX_WORKERS", "TEMPERATURE"} & set(env)
    # локальная qwen3 тратит лимит на рассуждения: у baseline свой лимит, не агента (2048) и не 1024
    assert env["MAX_TOKENS"] == "4096"
    assert env["LLM_BASE_URL"] == "http://o/v1" and env["DATA_DIR"] == str(Path("/tmp/x"))
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_review_baseline_runs_in_batches_and_caches_each(tmp_path):
    from judge import BASELINE_BATCH

    many = [_case(f"own-{i}", question=f"Вопрос {i}?") for i in range(BASELINE_BATCH + 3)]
    sizes = []

    def runner(qs, d):
        sizes.append(len(qs))
        if len(sizes) == 2:
            raise KeyboardInterrupt  # прогон оборвался на второй пачке
        return {q[0]: "x" for q in qs}

    cache = tmp_path / "base.csv"
    with pytest.raises(KeyboardInterrupt):
        answer_baseline(many, cache, tmp_path, "m", runner=runner)
    assert sizes == [BASELINE_BATCH, 3]
    done = []
    answer_baseline(many, cache, tmp_path, "m", runner=lambda qs, d: done.append(len(qs)) or {q[0]: "y" for q in qs})
    assert done == [3]  # первая пачка уже в кеше


def test_review_judge_client_has_own_timeout_and_no_agent_extra_body(monkeypatch):
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"reasoning_effort": "none"}')
    monkeypatch.setenv("LLM_TIMEOUT", "90")
    monkeypatch.delenv("JUDGE_TIMEOUT", raising=False)
    c = judge_client_from_env()
    assert c.s.timeout == 600.0 and c.s.extra_body == {}
    monkeypatch.setenv("JUDGE_TIMEOUT", "300")
    assert judge_client_from_env().s.timeout == 300.0


def test_review_parse_round_salvages_broken_json():
    raw = '{"a_fabricated": false, "b_fabricated": false, "reasoning": "ответ B "точнее"", "verdict": "B_better"}'
    r = parse_round(raw, "B")
    assert not r.error and r.verdict == "B_better"


def test_review_unload_model_stops_generator_and_embedder(monkeypatch):
    import judge

    stopped = []
    monkeypatch.setattr(judge.subprocess, "run", lambda args, **kw: stopped.append(args[-1]))
    judge.unload_model(["qwen3:4b", "bge-m3"], "http://localhost:11434/v1")
    assert stopped == ["qwen3:4b", "bge-m3"]
    stopped.clear()
    judge.unload_model(["qwen3:4b"], "https://api.example/v1")
    assert stopped == []


@needs_data
def test_review_bad_ref_title_named_docs_and_article_first_order(kb):
    assert "bad_ref" not in check_answer(
        "По п. 4.2 договора П-17/2024 и п. 5.4 Регламента согласования договоров нужна проверка.", kb, _case())
    assert "bad_ref" not in check_answer(
        "По п. 4.1 договора П-17/2024 и п. 2 Дополнительного соглашения № 1 адрес изменён.", kb, _case())
    cites = [{"doc_id": "nk_164"}, {"doc_id": "contract_p-17-2024"}]
    assert "bad_ref" not in check_answer(
        "По договору П-17/2024 оплата безналичная (п. 4.1); ставка НДС — НК РФ, ст. 164, п. 3.", kb, _case(),
        citations=cites)
    # настоящий несуществующий пункт договора по-прежнему ловится
    assert "bad_ref" in check_answer("По договору аренды № А-08/2024 плата вносится по п. 9.9.", kb, _case())


def test_baseline_max_tokens_override():
    from judge import baseline_env

    env = baseline_env({"MAX_TOKENS": "2048", "BASELINE_MAX_TOKENS": "8192"}, Path("/tmp/x"))
    assert env["MAX_TOKENS"] == "8192"


class _StubAssistant:
    def __init__(self, kb):
        self.kb, self.client = kb, None

    def answer(self, question):
        return {"answer": gen.FAIL_ANSWER, "status": "error", "citations": []}


@needs_data
def test_stub_scores_zero_without_judge(tmp_path, kb):
    # Заглушка не может выиграть: судья qwen3 путал стороны и ставил ей 1.0
    data, out = _data_dir(tmp_path), tmp_path / "out"
    fj = FakeJudgeClient(_verdict("B_much_better"))  # раунд 1: наш — B, «намного лучше»
    m = run("t3", data, out, own_path=tmp_path / "none.csv", assistant=_StubAssistant(kb),
            judge=Judge(fj, out / "judge_cache.jsonl", "m"), baseline_runner=_base)
    assert fj.calls == 0 and m["match_rate"] == 0.0 and m["n"] == 2
    assert "заглушка" in (out / "t3" / "report.md").read_text(encoding="utf-8")
