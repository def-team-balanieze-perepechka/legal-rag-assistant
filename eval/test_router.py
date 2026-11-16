from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT))

import prompts  # noqa: E402
from router import Plan, build_catalog, parse_plan, route, router_enabled  # noqa: E402
from test_agent import FakeClient, kb  # noqa: E402,F401


def plan_json(**kw) -> str:
    import json
    base = {"scenario": "fact", "docs": [], "queries": [], "event_date": None, "claims": []}
    base.update(kw)
    return json.dumps(base, ensure_ascii=False)



def test_catalog_has_line_per_document(kb):
    lines = build_catalog(kb).splitlines()
    assert len(lines) == len(kb.documents) == 21
    for line in lines:
        cols = line.split(" | ")
        assert cols[0] in kb.documents
        assert cols[1] == kb.documents[cols[0]].short_ref


def test_parse_plan_valid():
    raw = ('{"scenario":"date_edition","docs":["a","zzz"],'
           '"queries":["q1"],"event_date":"2024-05-01","claims":["c"]}')
    assert parse_plan(raw, {"a"}) == Plan("date_edition", ["a"], ["q1"], date(2024, 5, 1), ["c"])


@pytest.mark.parametrize("raw", ["", "не json", "[1,2]"])
def test_parse_plan_garbage_is_none(raw):
    assert parse_plan(raw, {"a"}) is None


def test_parse_plan_empty_object_is_fact():
    assert parse_plan("{}", {"a"}) == Plan("fact", [], [], None, [])


def test_parse_plan_unknown_scenario_is_fact():
    assert parse_plan(plan_json(scenario="weird"), set()).scenario == "fact"


def test_parse_plan_limits():
    ids = {f"d{i}" for i in range(6)}
    p = parse_plan(plan_json(docs=[f"d{i}" for i in range(6)],
                             queries=["q"] * 4 + ["x" * 300],
                             claims=["c1", "c2", "c3", "c4", "c5"]), ids)
    assert len(p.docs) == 4 and len(p.claims) == 3
    assert len(p.queries) == 3
    p = parse_plan(plan_json(queries=["x" * 300]), set())
    assert p.queries == ["x" * 200]


def test_parse_plan_bad_types():
    raw = ('{"scenario":"fact","docs":"a",'
           '"claims":5,"queries":null,"event_date":20240501}')
    assert parse_plan(raw, {"a"}) == Plan("fact", [], [], None, [])
    assert parse_plan(plan_json(event_date="не дата"), set()).event_date is None


def test_parse_plan_accepts_fenced_json():
    raw = "Вот план:\n```json\n" + plan_json(scenario="illegal") + "\n```"
    assert parse_plan(raw, set()).scenario == "illegal"


def test_route_returns_plan_and_sends_history_and_catalog(kb):
    client = FakeClient(reply=plan_json(docs=["nk_164"]))
    plan = route(client, kb, "вопрос", [{"role": "user", "content": "прошлая реплика"}])
    assert plan.docs == ["nk_164"]
    assert client.max_tokens == [2048]
    content = client.messages[0][-1]["content"]
    assert "прошлая реплика" in content and "вопрос" in content and "nk_164 | " in content


@pytest.mark.parametrize("reply", [RuntimeError("down"), ""])
def test_route_failure_returns_none_and_logs(kb, capsys, reply):
    assert route(FakeClient(reply=reply), kb, "вопрос", None) is None
    assert "[router]" in capsys.readouterr().err


def test_router_enabled_flag(monkeypatch):
    monkeypatch.delenv("ROUTER", raising=False)
    assert router_enabled() is True  # На платформе переменных нет: роутер включен
    monkeypatch.setenv("ROUTER", "")
    assert router_enabled() is True
    for off in ("0", "false", "no", "off"):
        monkeypatch.setenv("ROUTER", off)
        assert router_enabled() is False
    monkeypatch.setenv("ROUTER", "1")
    assert router_enabled() is True


def _plan(scenario, **kw):
    return Plan(scenario, kw.get("docs", []), kw.get("queries", []),
                kw.get("event_date"), kw.get("claims", []))


def test_plan_hint_per_scenario():
    assert prompts.plan_hint(_plan("fact"), []) == ""
    h = prompts.plan_hint(_plan("false_premise", claims=["НДС 20%"]), [])
    assert "НДС 20%" in h and "false_premise" in h
    h = prompts.plan_hint(_plan("date_edition", event_date=date(2024, 5, 1)), [])
    assert "01.05.2024" in h and "действовала на дату события" in h
    h = prompts.plan_hint(_plan("multi_doc"), ["регламент А", "письмо К-12"])
    assert "регламент А" in h and "письмо К-12" in h
    assert "refuse_illegal" in prompts.plan_hint(_plan("illegal"), [])
    assert "прежней" in prompts.plan_hint(_plan("old_law"), [])
    assert "offtopic" in prompts.plan_hint(_plan("offtopic"), [])
    assert "insufficient" in prompts.plan_hint(_plan("not_in_kb"), [])


@pytest.mark.parametrize("scenario", ["false_premise", "illegal", "not_in_kb", "old_law", "multi_doc"])
def test_plan_hint_keeps_answer_to_the_question(scenario):
    h = prompts.plan_hint(_plan(scenario, claims=["c"]), [])
    assert "не добавляй" in h.lower() and "сверх вопроса" in h


def test_plan_hint_omits_empty_parts():
    for p in (_plan("false_premise"), _plan("date_edition"), _plan("multi_doc")):
        h = prompts.plan_hint(p, [])
        assert "{" not in h and "None" not in h and "«»" not in h and "«»" not in h


def _user_message(client):
    return client.messages[0][-1]["content"]


def test_generate_without_plan_prompt_identical_to_v2a(kb):
    import generate as gen
    chunks = kb.chunks[:2]
    client = FakeClient(reply='{"status":"answer","answer":"ок","citations":[],"missing_info":""}')
    gen.generate(client, kb, "вопрос?", [], chunks)
    expected = prompts.USER.format(history="", question="вопрос?",
                                   fragments=gen.format_fragments(chunks, kb), extra="")
    assert _user_message(client) == expected
    assert "ПЛАН" not in expected


def test_generate_with_plan_appends_hint(kb):
    import generate as gen
    chunks = kb.chunks[:2]
    client = FakeClient(reply='{"status":"refuse_illegal","answer":"нет","citations":[],"missing_info":""}')
    gen.generate(client, kb, "вопрос?", [], chunks, plan=_plan("illegal"))
    msg = _user_message(client)
    assert "refuse_illegal" in msg
    assert msg.index("refuse_illegal") > msg.index(chunks[-1].chunk_id)
    assert client.messages[0][0]["content"] == prompts.system_prompt(False)


from search import Retriever  # noqa: E402
import generate as gen_module  # noqa: E402

VAT_Q = "Какая общая ставка НДС, если льготные ставки не применяются?"


def _retriever(kb, **kw):
    r = Retriever(kb, FakeClient(**kw))
    r.build_dense()
    return r


def _ids(hits):
    return [h.chunk.chunk_id for h in hits]


def _doc_ids(hits):
    return {h.chunk.doc_id for h in hits}


@pytest.fixture(scope="module")
def retriever(kb):
    return _retriever(kb)


def test_plan_small_doc_included_whole(kb, retriever):
    plan = _plan("multi_doc", docs=["memo_yuo-7"])
    got = set(_ids(retriever.search("вопрос про налог", plan=plan)))
    assert {kb.chunks[i].chunk_id for i in retriever.by_doc["memo_yuo-7"]} <= got


def test_plan_large_doc_gives_four_fragments(kb, retriever):
    q = "Как считается налог на прибыль организаций?"
    base = set(_ids(retriever.search(q)))
    plan = _plan("old_law", docs=["nk_284"], queries=["налог на прибыль ставка"])
    added = set(_ids(retriever.search(q, plan=plan))) - base
    nk284 = {c for c in added if c.startswith("nk_284#")}
    assert 1 <= len(nk284) <= 4


def test_plan_cap_20_chunks(kb, retriever):
    q = "общий вопрос"
    base = set(_ids(retriever.search(q)))
    plan = _plan("multi_doc", docs=["nk_164", "nk_284", "nk_145", "koap_13-11"])
    added = set(_ids(retriever.search(q, plan=plan))) - base
    assert 0 < len(added) <= 20


def test_plan_without_embeddings_does_not_fail(kb):
    r = _retriever(kb, fail_embed=True, fail_rerank=True)
    got = _ids(r.search("ставка НДС", plan=_plan("multi_doc", docs=["nk_164"], queries=["ставка НДС"])))
    assert any(c.startswith("nk_164#") for c in got)


@pytest.mark.parametrize("scenario", ["not_in_kb", "fact", "illegal"])
def test_plan_empty_docs_and_queries_equals_v2a(retriever, scenario):
    q = "Какой срок приёмки по качеству?"
    assert _ids(retriever.search(q, plan=_plan(scenario))) == _ids(retriever.search(q))


@pytest.mark.parametrize("scenario", ["multi_doc", "date_edition", "old_law", "illegal"])
def test_plan_docs_added_for_expanding_scenarios(kb, retriever, scenario):
    plan = _plan(scenario, docs=["memo_yuo-7"])
    got = set(_ids(retriever.search("вопрос про налог", plan=plan)))
    assert {kb.chunks[i].chunk_id for i in retriever.by_doc["memo_yuo-7"]} <= got


@pytest.mark.parametrize("scenario", ["fact", "false_premise", "not_in_kb", "offtopic"])
def test_plan_docs_ignored_for_other_scenarios(retriever, scenario):
    """Документы плана расширяют выдачу только для multi_doc / date_edition / old_law."""
    q = "Какой срок приёмки по качеству?"
    plan = _plan(scenario, docs=["memo_yuo-7", "reglament_soglasovaniya_2025"])  # Без закона: фильтр договоров не включается
    assert _ids(retriever.search(q, plan=plan)) == _ids(retriever.search(q))


def test_plan_queries_add_rankings(kb, retriever):
    q = "Что-то про документы"
    plan = _plan("multi_doc", queries=["персональные данные оператор согласие субъекта"])
    assert any(c.startswith("152fz_9#") for c in _ids(retriever.search(q, plan=plan)))


def test_law_beats_contract_when_no_entity(retriever):
    plan = _plan("false_premise", docs=["nk_164"], queries=["общая налоговая ставка НДС 20 процентов"])
    hits = retriever.search(VAT_Q, plan=plan)
    assert "nk_164#p3" in _ids(hits)
    assert "contract_p-17-2024" not in _doc_ids(hits)


def test_named_contract_not_dropped_by_law_rule(retriever):
    plan = _plan("false_premise", docs=["nk_164"], queries=["общая налоговая ставка НДС"])
    hits = retriever.search("Какой НДС указан в договоре П-17/2024?", plan=plan)
    assert "contract_p-17-2024" in _doc_ids(hits)


def test_plan_with_contract_doc_keeps_contracts(retriever):
    plan = _plan("multi_doc", docs=["nk_164", "contract_p-17-2024"])
    assert "contract_p-17-2024" in _doc_ids(retriever.search(VAT_Q, plan=plan))


def test_event_date_label(kb):
    c23 = [c for c in kb.chunks if c.doc_id == "reglament_soglasovaniya_2023"][:1]
    c25 = [c for c in kb.chunks if c.doc_id == "reglament_soglasovaniya_2025"][:1]
    law = [c for c in kb.chunks if c.doc_id == "tk_70"][:1]
    d = date(2024, 5, 1)
    assert "действовала на дату события" in gen_module.format_fragments(c23, kb, d)
    assert "на дату события не действовала" in gen_module.format_fragments(c25, kb, d)
    assert "дату события" not in gen_module.format_fragments(law, kb, d)
    assert "дату события" not in gen_module.format_fragments(c23, kb)


from types import SimpleNamespace  # noqa: E402

from checks import drop_invented_articles, drop_leaked_chunk_ids, missing_article_note  # noqa: E402


def _fake_chunk(doc_id, text):
    return SimpleNamespace(doc_id=doc_id, text=text, ref="", chunk_id=f"{doc_id}#x")


def test_missing_articles_152fz_9(kb):
    chunks = [c for c in kb.chunks if c.doc_id == "152fz_9"]
    note = missing_article_note(chunks, kb)
    assert "6, 10, 11" in note and "152-ФЗ" in note
    assert " 9," not in note and "9 " not in note.split("152-ФЗ")[0]


def test_missing_articles_skips_other_codes(kb):
    assert missing_article_note([_fake_chunk("koap_13-11", "см. статьи 13.11 КоАП РФ.")], kb) == ""
    assert missing_article_note([_fake_chunk("tk_70", "по статье 5 Трудового кодекса")], kb) == ""


def test_missing_articles_none_when_all_present(kb):
    assert missing_article_note([_fake_chunk("tk_70", "в соответствии со статьей 59 настоящего Кодекса")], kb) == ""


def test_invented_article_removed_from_refusal(kb):
    text = "В базе нет сведений. Применяется ст. 260 закона. Уточните вопрос."
    assert drop_invented_articles(text, "insufficient", [], kb) == "В базе нет сведений. Уточните вопрос."


def test_invented_article_kept_for_answer_status(kb):
    text = "В базе нет сведений. Применяется ст. 260 закона. Уточните вопрос."
    assert drop_invented_articles(text, "answer", [], kb) == text


def test_invented_article_kept_if_in_kb_or_fragments(kb):
    text = "Данных мало. Смотрите ст. 164 НК РФ. Ещё статья 777 упомянута."
    chunks = [_fake_chunk("tk_70", "см. статью 777")]
    assert drop_invented_articles(text, "insufficient", chunks, kb) == text


@pytest.mark.parametrize("status,expected", [
    ("insufficient", "В базе знаний нет сведений для ответа на этот вопрос."),
    ("refuse_illegal", "Я не могу помочь с этим запросом."),
    ("offtopic", "Я отвечаю только на вопросы по базе знаний юридического отдела."),
])
def test_all_sentences_invented_gives_neutral_refusal(kb, status, expected):
    text = "Согласно ст. 260 это запрещено. По ст. 261 тоже нельзя."
    assert drop_invented_articles(text, status, [], kb) == expected


def test_single_invented_sentence_is_replaced_not_kept(kb):
    out = drop_invented_articles("Согласно ст. 260 это запрещено.", "insufficient", [], kb)
    assert "260" not in out and out


def test_empty_answer_is_left_alone(kb):
    assert drop_invented_articles("", "insufficient", [], kb) == ""


def test_abbreviation_does_not_split_sentence(kb):
    text = "Нужны сведения. По п. 5 ст. 260 вопрос не решить. Обратитесь к юристу."
    assert drop_invented_articles(text, "refuse_illegal", [], kb) == "Нужны сведения. Обратитесь к юристу."


def test_leaked_chunk_id_removed(kb):
    assert drop_leaked_chunk_ids("Ставка 20% (memo_yuo-7#whole) по закону.", kb) == "Ставка 20% по закону."
    assert drop_leaked_chunk_ids("Срок (п. 3.1) пять дней.", kb) == "Срок (п. 3.1) пять дней."


def test_generate_applies_checks_only_with_plan(kb):
    reply = ('{"status":"answer","answer":"Ставка 20% (memo_yuo-7#whole) по закону."'
             ',"citations":[],"missing_info":""}')
    chunks = [c for c in kb.chunks if c.doc_id == "152fz_9"][:3]
    c1 = FakeClient(reply=reply)
    r1 = gen_module.generate(c1, kb, "вопрос?", [], chunks)
    assert "memo_yuo-7#whole" in r1["answer"] and "в базе нет" not in _user_message(c1)
    c2 = FakeClient(reply=reply)
    r2 = gen_module.generate(c2, kb, "вопрос?", [], chunks, plan=_plan("fact"))
    assert "memo_yuo-7#whole" not in r2["answer"]
    assert "в базе нет" in _user_message(c2)



import json as _json  # noqa: E402

from assistant import Assistant  # noqa: E402

ANSWER = _json.dumps({"status": "answer", "answer": "Ответ.", "citations": [], "missing_info": ""},
                     ensure_ascii=False)


def _router_client(plan_reply, **kw):
    def reply(messages):
        if messages[0]["content"] == prompts.ROUTER_SYSTEM:
            if isinstance(plan_reply, Exception):
                raise plan_reply
            return plan_reply
        return ANSWER
    return FakeClient(reply=reply, **kw)


def _assistant(kb, client):
    a = Assistant(kb, client)
    a.build_index()
    return a


def test_answer_with_router_makes_two_chat_calls_and_records_plan(kb, monkeypatch):
    monkeypatch.setenv("ROUTER", "1")
    client = _router_client(plan_json(scenario="illegal", docs=["koap_13-11"]))
    res = _assistant(kb, client).answer("Как скрыть утечку персональных данных?", [])
    assert client.calls["chat"] == 2
    assert res["debug"]["plan"]["scenario"] == "illegal"
    assert res["debug"]["plan"]["docs"] == ["koap_13-11"]
    assert "refuse_illegal" in client.messages[1][-1]["content"]


def test_router_failure_falls_back_to_v2a(kb, monkeypatch):
    q = "Какой срок приёмки по качеству?"
    monkeypatch.setenv("ROUTER", "1")
    failed = _assistant(kb, _router_client(RuntimeError("down"))).answer(q, [])
    monkeypatch.setenv("ROUTER", "0")
    off = _assistant(kb, _router_client(plan_json())).answer(q, [])
    assert failed["debug"]["plan"] is None
    assert failed["debug"]["retrieved"] == off["debug"]["retrieved"]


def test_router_off_is_v2a(kb, monkeypatch):
    monkeypatch.setenv("ROUTER", "0")
    client = _router_client(plan_json(scenario="illegal"))
    res = _assistant(kb, client).answer("Какой срок приёмки?", [])
    assert client.calls["chat"] == 1
    assert res["debug"]["plan"] is None
    assert "ПЛАН" not in client.messages[0][-1]["content"]


def test_answer_contract_unchanged(kb):
    res = _assistant(kb, _router_client(plan_json())).answer("Какой срок приёмки?", [])
    assert {"status", "answer", "citations", "missing_info", "steps", "debug"} <= set(res)
    assert {"retrieved", "seconds", "plan"} <= set(res["debug"])


def test_router_sees_history_in_dialog(kb, monkeypatch):
    monkeypatch.setenv("ROUTER", "1")
    client = _router_client(plan_json())
    dialog = "Пользователь: Что с договором П-17/2024?\nАссистент: Уточните.\nПользователь: Какой срок оплаты?"
    _assistant(kb, client).answer(dialog)
    msg = client.messages[0][-1]["content"]
    assert "Что с договором П-17/2024?" in msg and msg.rstrip().endswith("Верни JSON-план.")
    assert "ВОПРОС ПОЛЬЗОВАТЕЛЯ:\nКакой срок оплаты?" in msg


def _route_with(kb, **kw):
    return route(FakeClient(reply=plan_json(**kw)), kb, "вопрос", None)


def test_route_fact_with_claims_becomes_false_premise(kb):
    p = _route_with(kb, scenario="fact", claims=["общая ставка НДС 20%"])
    assert p.scenario == "false_premise" and p.claims == ["общая ставка НДС 20%"]


def test_route_fact_without_claims_stays_fact(kb):
    assert _route_with(kb, scenario="fact").scenario == "fact"


def test_route_claims_do_not_override_other_scenarios(kb):
    assert _route_with(kb, scenario="illegal", claims=["c"]).scenario == "illegal"


def test_route_date_edition_with_only_laws_becomes_old_law(kb):
    p = _route_with(kb, scenario="date_edition", docs=["mrot_1"], event_date="2024-12-31")
    assert p.scenario == "old_law"
    assert p.event_date is not None  # дата остаётся в плане


def test_route_date_edition_with_company_doc_stays(kb):
    p = _route_with(kb, scenario="date_edition", docs=["mrot_1", "reglament_soglasovaniya_2023"])
    assert p.scenario == "date_edition"
    assert _route_with(kb, scenario="date_edition").scenario == "date_edition"


def test_router_prompt_states_the_scenario_rules():
    sp = prompts.ROUTER_SYSTEM
    assert "только для документов компании" in sp           # date_edition vs old_law
    assert "docs пустым" in sp                              # not_in_kb: не подбирать похожие
    assert "нескольких частей" in sp                        # двусоставной вопрос -> multi_doc
    assert "ответственност" in sp                           # illegal: документ об ответственности
    assert "договор — только" in sp                         # договор только если назван


def test_route_fact_with_documents_of_two_types_becomes_multi_doc(kb):
    p = _route_with(kb, scenario="fact", docs=["reglament_soglasovaniya_2025", "letter_k-12"])
    assert p.scenario == "multi_doc" and p.docs == ["reglament_soglasovaniya_2025", "letter_k-12"]


def test_route_fact_with_documents_of_one_type_stays_fact(kb):
    assert _route_with(kb, scenario="fact", docs=["nk_145", "nk_164"]).scenario == "fact"


def test_route_claims_win_over_multi_doc_rule(kb):
    p = _route_with(kb, scenario="fact", docs=["reglament_soglasovaniya_2025", "letter_k-12"], claims=["c"])
    assert p.scenario == "false_premise"


def test_router_prompt_warns_against_picking_documents_from_same_code():
    assert "из того же кодекса" in prompts.ROUTER_SYSTEM


def test_multi_doc_hint_uses_current_edition_without_date():
    h = prompts.plan_hint(_plan("multi_doc"), ["регламент"])
    assert "нет даты" in h and "действующую редакцию" in h


def _route_q(kb, question, **kw):
    return route(FakeClient(reply=plan_json(**kw)), kb, question, None)


def test_unknown_company_becomes_not_in_kb(kb):
    p = _route_q(kb, "Какой срок оплаты по договору с ООО «Бета-Логистик»?",
                 scenario="fact", docs=["contract_p-17-2024"], queries=["срок оплаты"])
    assert p.scenario == "not_in_kb" and p.docs == [] and p.queries == ["срок оплаты"]


@pytest.mark.parametrize("scenario", ["fact", "false_premise", "multi_doc", "date_edition"])
def test_unknown_company_overrides_content_scenarios(kb, scenario):
    q = "Что сказано в договоре с АО «Гамма»?"
    assert _route_q(kb, q, scenario=scenario).scenario == "not_in_kb"


@pytest.mark.parametrize("question", [
    "Какая арендная плата по договору с ООО «Северный Склад»?",
    "Срок оплаты по договору с ООО «Альфа-Трейдинга» — 30 дней?",
    "Какой срок оплаты по договору с ООО «Альфа-Трейд»?",
    "Что ООО «Меридиан» обязано сделать по регламенту?",
    "Какой срок оплаты по договору П-17/2024?",
    "Какая общая ставка НДС?",
])
def test_known_or_no_company_keeps_scenario(kb, question):
    assert _route_q(kb, question, scenario="fact").scenario == "fact"


def test_unknown_company_does_not_override_illegal_or_offtopic(kb):
    q = "Как скрыть сделку с ООО «Бета-Логистик» от налоговой?"
    assert _route_q(kb, q, scenario="illegal").scenario == "illegal"
    assert _route_q(kb, "Закажите пиццу в ООО «Додо»", scenario="offtopic").scenario == "offtopic"


from llm import Reply  # noqa: E402


def _cut():
    return Reply("<think>Пользователь спрашивает про ставку, нужно подумать", "length")


def test_route_retries_with_bigger_limit_when_cut_by_length(kb):
    client = FakeClient(reply=[_cut(), Reply(plan_json(scenario="illegal"), "stop")])
    plan = route(client, kb, "вопрос", None)
    assert plan is not None and plan.scenario == "illegal"
    assert client.max_tokens == [2048, 4096] and client.calls["chat"] == 2


def test_route_gives_up_after_one_retry(kb, capsys):
    client = FakeClient(reply=[_cut(), _cut(), _cut()])
    assert route(client, kb, "вопрос", None) is None
    assert client.calls["chat"] == 2  # одна попытка и один повтор, не больше
    assert "[router]" in capsys.readouterr().err


def test_route_does_not_retry_a_finished_garbage_reply(kb):
    client = FakeClient(reply=[Reply("не JSON", "stop"), Reply(plan_json(), "stop")])
    assert route(client, kb, "вопрос", None) is None
    assert client.calls["chat"] == 1


def test_route_adds_no_think_when_flag_set(kb, monkeypatch):
    monkeypatch.setenv("LLM_NO_THINK", "1")
    client = FakeClient(reply=plan_json())
    route(client, kb, "вопрос", None)
    assert client.messages[0][-1]["content"].endswith(prompts.NO_THINK)
    monkeypatch.delenv("LLM_NO_THINK")
    client = FakeClient(reply=plan_json())
    route(client, kb, "вопрос", None)
    assert prompts.NO_THINK not in client.messages[0][-1]["content"]


@pytest.mark.parametrize("question", [
    "Что это за договор поставки П-17/2024?",
    "Какой срок оплаты по договору с ООО «Северный Склад»?",
    "Что сказано в договоре с Альфа-Трейдингом о качестве?",
])
def test_known_contract_or_company_is_never_not_in_kb(kb, question):
    p = _route_q(kb, question, scenario="not_in_kb", docs=["contract_p-17-2024"])
    assert p.scenario == "fact"


def test_known_entity_with_claims_becomes_false_premise(kb):
    p = _route_q(kb, "По договору П-17/2024 оплата за 30 дней, верно?",
                 scenario="not_in_kb", claims=["оплата за 30 дней"])
    assert p.scenario == "false_premise"


def test_not_in_kb_stays_when_no_known_entity(kb):
    assert _route_q(kb, "Сколько дней ежегодного оплачиваемого отпуска положено работнику?",
                    scenario="not_in_kb").scenario == "not_in_kb"
    assert _route_q(kb, "Какой срок оплаты по договору с ООО «Бета-Логистик»?",
                    scenario="fact").scenario == "not_in_kb"  # неизвестная организация — по-прежнему not_in_kb


@pytest.mark.parametrize("question", ["", "   ", "\n\t", "???", " . "])
def test_empty_question_gets_polite_answer_without_any_call(kb, question):
    client = _router_client(plan_json(scenario="illegal"))
    res = _assistant(kb, client).answer(question, [])
    assert res["status"] == "offtopic" and "Вопрос не получен" in res["answer"]
    assert res["citations"] == [] and res["steps"] == []
    assert res["debug"]["retrieved"] == [] and res["debug"]["plan"] is None
    assert client.calls["chat"] == 0 and client.calls["rerank"] == 0


def test_empty_question_asks_to_clarify_when_allowed(kb):
    client = _router_client(plan_json())
    res = _assistant(kb, client).answer("  ", [], allow_clarify=True)
    assert res["status"] == "clarify" and client.calls["chat"] == 0


def test_empty_last_turn_in_dialog_is_empty_question(kb):
    client = _router_client(plan_json())
    res = _assistant(kb, client).answer("Пользователь: ???")
    assert res["status"] == "offtopic" and client.calls["chat"] == 0


def test_short_real_question_is_not_blocked(kb, monkeypatch):
    monkeypatch.setenv("ROUTER", "0")
    client = _router_client(plan_json())
    res = _assistant(kb, client).answer("НДС?", [])
    assert client.calls["chat"] == 1 and "Вопрос не получен" not in res["answer"]
