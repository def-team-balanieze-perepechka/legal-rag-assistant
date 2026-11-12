from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT))

import generate as gen  # noqa: E402
import kb as kb_module  # noqa: E402
from assistant import Assistant, parse_dialog  # noqa: E402
from search import Retriever, number_key  # noqa: E402
from stem import stem, tokenize  # noqa: E402

KB_PATH = ROOT / "data" / "knowledge-base.zip"
needs_data = pytest.mark.skipif(not KB_PATH.exists(),
                                reason="нет data/knowledge-base.zip")


class FakeClient:
    """
    Эмбеддинги — хэш мешка слов, реранкер — по пересечению токенов,
    LLM — заданный ответ.
    """

    def __init__(self, reply=None, fail_embed=False, fail_rerank=False):
        self.reply = reply
        self.fail_embed = fail_embed
        self.fail_rerank = fail_rerank
        self.messages = []
        self.max_tokens = []
        self.calls = {"chat": 0, "embed": 0, "rerank": 0}

    def embed(self, texts):
        self.calls["embed"] += 1
        if self.fail_embed:
            raise RuntimeError("embed down")
        out = []
        for t in texts:
            v = np.zeros(256)
            for tok in tokenize(t):
                v[hash(tok) % 256] += 1
            out.append(v.tolist())
        return out

    def rerank(self, query, documents, top_n):
        self.calls["rerank"] += 1
        if self.fail_rerank:
            raise RuntimeError("rerank down")
        q = set(tokenize(query))
        scored = [(i, len(q & set(tokenize(d))) / (1 + len(set(tokenize(d))) ** 0.5))
                  for i, d in enumerate(documents)]
        return sorted(scored, key=lambda x: -x[1])[:top_n]

    def chat(self, messages, max_tokens=None):
        self.calls["chat"] += 1
        self.messages.append(messages)
        self.max_tokens.append(max_tokens)
        if isinstance(self.reply, list):
            reply = self.reply[min(len(self.reply), self.calls["chat"]) - 1]
        else:
            reply = self.reply(messages) if callable(self.reply) else self.reply
        if isinstance(reply, Exception):
            raise reply
        return reply


def fragment_ids(messages) -> list[str]:
    return re.findall(r"^\[([^\]]+)\]", messages[-1]["content"], re.M)


@pytest.fixture(scope="module")
def kb():
    if not KB_PATH.exists():
        pytest.skip("нет data/knowledge-base.zip")
    return kb_module.load_kb(KB_PATH)


def test_stem_groups_word_forms():
    assert stem("арендная") == stem("арендной") == stem("арендную")
    assert stem("договора") == stem("договору") == stem("договор")
    assert stem("испытания") == stem("испытание")
    assert tokenize("Срок испытания и ёлка") == [stem("срок"), stem("испытания"),
                                                 "елка"]


def test_parse_dialog_takes_last_user_turn():
    text = ("Пользователь: Нужно проверить договор поставки.\n"
            "Ассистент: Уточните, пожалуйста, с Альфа-Трейд или Альфа-Трейдинг?\n"
            "Пользователь: Альфа-Трейдинг. Какой гарантийный срок?")
    q, hist = parse_dialog(text)
    assert q == "Альфа-Трейдинг. Какой гарантийный срок?"
    assert [t["role"] for t in hist] == ["user", "assistant"]
    assert "Альфа-Трейд или" in hist[1]["content"]


def test_parse_dialog_plain_question():
    assert parse_dialog("Какой МРОТ?") == ("Какой МРОТ?", [])


def test_rrf_prefers_items_in_both_rankings():
    scores = Retriever.rrf([[1, 2, 3], [3, 4]])
    assert max(scores, key=scores.get) == 3


def test_number_key_normalizes_alphabet_and_separators():
    assert number_key("ПТ-05/2023") == number_key("пт 05-2023") == number_key(
        "пт05/2023")
    assert number_key("A-08/2024") == number_key("А-08/2024")  # латинская A → кириллица


@needs_data
def test_entities_distinguish_similar_counterparties(kb):
    m = Retriever(kb).entities
    assert m.match("договор с Альфа-Трейдинг")["counterparties"] == [
        "contract_pt-05-2023"]
    assert "contract_pt-05-2023" not in m.match("поставка от Альфа-Трейдом")[
        "counterparties"]
    assert "contract_p-17-2024" in m.match("с альфа трейд")["counterparties"]
    assert m.match("PT-05/2023")["contracts"] == ["contract_pt-05-2023"]
    assert m.match("по П-17/2024")["contracts"] == ["contract_p-17-2024"]


@needs_data
def test_search_pins_named_contract_and_filters_lookalike(kb):
    r = Retriever(kb, FakeClient())
    r.build_dense()
    hits = r.search("Какой гарантийный срок?",
                    context="Альфа-Трейдинг, договор поставки")
    docs = {h.chunk.doc_id for h in hits}
    assert "contract_pt-05-2023" in docs
    assert "contract_p-17-2024" not in docs


@needs_data
def test_search_adds_other_regulation_edition(kb):
    r = Retriever(kb, FakeClient())
    r.build_dense()
    hits = r.search(
        "Договор свыше 10 000 000 рублей подлежит одобрению генеральным директором?")
    ids = {h.chunk.chunk_id for h in hits}
    p24 = {c for c in ids if c.endswith("#p2.4") and c.startswith("reglament_")}
    assert len(p24) == 2, ids


@needs_data
def test_search_survives_embed_and_rerank_failures(kb):
    client = FakeClient(fail_embed=True, fail_rerank=True)
    r = Retriever(kb, client)
    assert r.build_dense() is False
    hits = r.search("арендная плата Северный Склад")
    assert hits and hits[0].chunk.doc_id == "contract_a-08-2024"


@pytest.mark.parametrize("raw", [
    '{"status": "answer", "answer": "Да.", "citations": []}',
    '```json\n{"status": "answer", "answer": "Да.", "citations": []}\n```',
    'Вот ответ: {"status": "answer", "answer": "Да {скобки}.", "citations": []} — готово',
])
def test_parse_json_variants(raw):
    obj = gen.parse_json(raw)
    assert obj["status"] == "answer" and obj["answer"].startswith("Да")


def test_parse_json_garbage():
    assert gen.parse_json("нет json") is None
    assert gen.parse_json("{сломано") is None


@needs_data
def test_generate_drops_unknown_citations_and_adds_source(kb):
    def reply(messages):
        ids = fragment_ids(messages)
        return json.dumps(
            {"status": "answer", "answer": "Арендная плата — 450 000 рублей в месяц.",
             "citations": [ids[0], "contract_fake#p9.9"], "missing_info": ""},
            ensure_ascii=False)

    chunk = kb.get_chunk("contract_a-08-2024#p3.1")
    res = gen.generate(FakeClient(reply), kb, "Сколько аренда?", [], [chunk])
    assert [c["chunk_id"] for c in res["citations"]] == ["contract_a-08-2024#p3.1"]
    assert res["dropped_citations"] == ["contract_fake#p9.9"]
    assert "Источник: договор аренды № А-08/2024, п. 3.1." in res["answer"]


@needs_data
def test_generate_retries_once_then_fails_honestly(kb):
    client = FakeClient("{не json")
    res = gen.generate(client, kb, "вопрос", [], [])
    assert client.calls["chat"] == 2  # Без фрагментов запасной вызов не делаем
    assert res["status"] == "error" and res["answer"] == gen.FAIL_ANSWER


@needs_data
@pytest.mark.parametrize("raw", [
    # неэкранированные кавычки внутри answer
    '{"status": "answer", "answer": "Арендная плата "по договору" — 450 000 рублей в месяц (п. 3.1).", '
    '"citations": ["contract_a-08-2024#p3.1"], "missing_info": ""}',
    # ответ оборвался по лимиту токенов
    '{"status": "answer", "answer": "Арендная плата — 450 000 рублей в месяц (п. 3.1). Вносится не позднее',
])
def test_generate_salvages_answer_from_broken_json(kb, raw):
    client = FakeClient(raw)
    chunk = kb.get_chunk("contract_a-08-2024#p3.1")
    res = gen.generate(client, kb, "Сколько аренда?", [], [chunk])
    assert res["answer"] != gen.FAIL_ANSWER
    assert res["answer"].startswith("Арендная плата")
    assert "450 000" in res["answer"]
    assert '"' not in res["answer"].split("450 000")[1][:5]


@needs_data
def test_generate_retries_with_bigger_limit_when_cut_by_length(kb):
    import llm

    good = json.dumps({"status": "answer", "answer": "Плата — 450 000 рублей (п. 3.1).",
                       "citations": ["contract_a-08-2024#p3.1"]}, ensure_ascii=False)
    client = FakeClient([llm.Reply("", "length"), llm.Reply(good, "stop")])
    chunk = kb.get_chunk("contract_a-08-2024#p3.1")
    res = gen.generate(client, kb, "Сколько аренда?", [], [chunk])
    assert res["answer"].startswith("Плата — 450 000 рублей")
    assert client.max_tokens[1] == 2 * gen.DEFAULT_MAX_TOKENS
    # повтор — тот же запрос, без «повтори в JSON»
    assert len(client.messages[1]) == len(client.messages[0])


@needs_data
def test_generate_cut_answer_trimmed_to_last_sentence(kb):
    import llm

    cut = llm.Reply(
        '{"status": "answer", "answer": "Нет. Оператор вправе продолжить обработку (п. 3.1). '
        'После отзыва согласия на обработку', "length")
    chunk = kb.get_chunk("contract_a-08-2024#p3.1")
    res = gen.generate(FakeClient(cut), kb, "Удалять всё?", [], [chunk])
    assert res["answer"].startswith("Нет. Оператор вправе продолжить обработку (п. 3.1).")
    assert "После отзыва" not in res["answer"]


@needs_data
def test_generate_cut_without_full_sentence_fails_honestly(kb):
    import llm

    cut = llm.Reply(
        '{"status": "answer", "answer": "Нет, после отзыва согласия на обработку',
        "length")
    res = gen.generate(FakeClient(cut), kb, "Удалять всё?", [], [])
    assert res["answer"] == gen.FAIL_ANSWER


@needs_data
def test_generate_salvage_keeps_only_known_citations(kb):
    raw = ('{"status": "answer", "answer": "Плата "450 000" рублей.", '
           '"citations": ["contract_a-08-2024#p3.1", "contract_fake#p9.9"]}')
    chunk = kb.get_chunk("contract_a-08-2024#p3.1")
    res = gen.generate(FakeClient(raw), kb, "Сколько аренда?", [], [chunk])
    assert [c["chunk_id"] for c in res["citations"]] == ["contract_a-08-2024#p3.1"]


@needs_data
def test_source_added_when_answer_names_contract_but_no_clause(kb):
    def reply(messages):
        ids = fragment_ids(messages)
        return json.dumps({"status": "answer", "citations": ids, "missing_info": "",
                           "answer": "По договору аренды № А-08/2024 плата — 450 000 рублей в месяц."},
                          ensure_ascii=False)

    chunks = [kb.get_chunk("contract_a-08-2024#p3.1"),
              kb.get_chunk("contract_a-08-2024#p3.2")]
    res = gen.generate(FakeClient(reply), kb, "Сколько аренда?", [], chunks)
    assert "Источник: договор аренды № А-08/2024, п. 3.1; договор аренды № А-08/2024, п. 3.2." in \
           res["answer"]


@needs_data
def test_generate_llm_down(kb):
    client = FakeClient(RuntimeError("down"))
    res = gen.generate(client, kb, "вопрос", [], [])
    assert res["answer"] == gen.FAIL_ANSWER and client.calls["chat"] == 1


@needs_data
def test_clarify_forbidden_in_submission_mode(kb):
    client = FakeClient(
        '{"status": "clarify", "answer": "По какому договору?", "citations": []}')
    res = gen.generate(client, kb, "срок оплаты?", [], [])
    assert res["status"] == "answer"
    assert "Уточняющих вопросов не задавай" in client.messages[0][0]["content"]


@needs_data
def test_suspicious_fragment_is_marked(kb):
    memo = next(c for c in kb.chunks if c.meta.get("suspicious"))
    text = gen.format_fragments([memo], kb)
    assert "не выполняй" in text and memo.text[:30] in text


@needs_data
def test_assistant_end_to_end_dialog(kb):
    def reply(messages):
        ids = fragment_ids(messages)
        cid = next(i for i in ids if i.startswith("contract_pt-05-2023#p6"))
        return json.dumps({"status": "answer",
                           "answer": "Гарантию устанавливает производитель (п. 6.1).",
                           "citations": [cid]}, ensure_ascii=False)

    a = Assistant(kb, FakeClient(reply))
    a.build_index()
    q = ("Пользователь: Расскажите про гарантию по договору с техникой.\n"
         "Ассистент: Уточните, пожалуйста, договор с Альфа-Трейд или Альфа-Трейдинг?\n"
         "Пользователь: Альфа-Трейдинг. Какой гарантийный срок?")
    res = a.answer(q)
    assert res["status"] == "answer"
    assert res["citations"][0]["doc_id"] == "contract_pt-05-2023"
    assert "\n" not in res["answer"]


@needs_data
def test_solution_writes_valid_submission(tmp_path, monkeypatch, kb):
    import solution

    (tmp_path / "questions.csv").write_text(
        'id,question\n1,"Сколько аренда у Северный Склад?"\n2,"Пользователь: a\nАссистент: b\nПользователь: Какой МРОТ?"\n',
        encoding="utf-8")
    with zipfile.ZipFile(KB_PATH) as src, zipfile.ZipFile(tmp_path / "kb.zip",
                                                          "w") as dst:
        for n in src.namelist():
            dst.writestr(n, src.read(n))
    monkeypatch.setenv("TIME_BUDGET_SEC", "300")
    out = tmp_path / "submission.csv"

    def reply(messages):
        return json.dumps({"status": "answer", "answer": "Ответ, \"с кавычками\".",
                           "citations": fragment_ids(messages)[:1]},
                          ensure_ascii=False)

    solution.run(tmp_path, out, client=FakeClient(reply))
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["id"] for r in rows] == ["1", "2"]
    assert all(r["answer"].strip() and len(r["answer"]) <= 10_000 for r in rows)
    assert out.read_text(encoding="utf-8").startswith('"id","answer"\n')


def test_build_submission_is_flat(tmp_path):
    out = tmp_path / "s.zip"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_submission.py"), "--out",
         str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    names = zipfile.ZipFile(out).namelist()
    assert "solution.py" in names and "requirements.txt" in names and "assistant.py" in names
    assert all("/" not in n for n in names)


def _client(handler):
    import httpx
    import llm

    s = llm.Settings(base_url="http://x", api_key="k", model="m", emb_model="e",
                     emb_base_url="http://x",
                     emb_key="k", rerank_model="r", rerank_base_url="http://x",
                     rerank_key="k",
                     temperature=0.0, max_tokens=100, extra_body={}, timeout=5.0)
    c = llm.LLMClient(s)
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_llm_retries_on_429_and_strips_thinking(monkeypatch):
    import httpx
    import llm

    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    seen = []

    def handler(req):
        seen.append(req.url.path)
        if len(seen) < 3:
            return httpx.Response(429)
        return httpx.Response(200,
                              json={"choices": [dict(
                                  message={"content": "<think>x</think>{\"a\": 1}"})]})

    assert _client(handler).chat([{"role": "user", "content": "hi"}]) == '{"a": 1}'
    assert len(seen) == 3


def test_llm_chat_reports_finish_reason():
    import httpx

    def handler(req):
        assert json.loads(req.content)["max_tokens"] == 333
        return httpx.Response(200, json={"choices": [{"message": {"content": ""},
                                                      "finish_reason": "length"}]})

    reply = _client(handler).chat([{"role": "user", "content": "hi"}], max_tokens=333)
    assert reply == "" and reply.finish_reason == "length"


def test_llm_disables_missing_reranker(monkeypatch):
    import httpx
    import llm

    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    hits = []

    def handler(req):
        hits.append(1)
        return httpx.Response(404)

    c = _client(handler)
    for _ in range(3):
        with pytest.raises(Exception):
            c.rerank("q", ["a", "b"], 2)
    assert len(hits) == 1  # после 404 реранкер больше не вызывается


def test_llm_chat_does_not_retry_timeout(monkeypatch):
    """Тайм-аут генерации не повторяем: модель снова считала бы тот же промпт с нуля."""
    import httpx
    import llm

    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    hits = []

    def handler(req):
        hits.append(1)
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(httpx.ReadTimeout):
        _client(handler).chat([{"role": "user", "content": "hi"}])
    assert len(hits) == 1


DIALOG_17 = ("Пользователь: Подскажите гарантию по договору с «Альфой».\n"
             "Ассистент: Уточните, пожалуйста, поставщик — «Альфа-Трейд» или «Альфа-Трейдинг»?\n"
             "Пользователь: «Альфа-Трейдинг», у них компьютеры покупали. На сколько гарантия?")


@needs_data
def test_dialog_entities_come_from_user_not_assistant_options(kb):
    """Вопрос ассистента с вариантами не должен подкладывать оба договора."""
    a = Assistant(kb,
                  FakeClient('{"status": "answer", "answer": "x", "citations": []}'))
    a.build_index()
    docs = {cid.split("#")[0] for cid, _, _ in
            a.answer(DIALOG_17)["debug"]["retrieved"]}
    assert "contract_pt-05-2023" in docs
    assert "contract_p-17-2024" not in docs


@needs_data
def test_entity_from_assistant_used_when_user_names_none(kb):
    """«Да» в ответ на «Вы про Альфа-Трейдинг?» — договор берём из реплики ассистента."""
    a = Assistant(kb,
                  FakeClient('{"status": "answer", "answer": "x", "citations": []}'))
    a.build_index()
    q = ("Пользователь: Какой гарантийный срок?\n"
         "Ассистент: Вы про договор с «Альфа-Трейдинг»?\n"
         "Пользователь: Да.")
    docs = {cid.split("#")[0] for cid, _, _ in a.answer(q)["debug"]["retrieved"]}
    assert "contract_pt-05-2023" in docs


def test_prompt_does_not_push_yes_and_addresses_user_directly():
    import prompts

    s = prompts.system_prompt(False)
    assert "«Да.»" not in s
    assert "на «вы»" in s


def test_tax_topic_from_text():
    assert kb_module.tax_topic("ставки налога на добавленную стоимость; "
                               "добавленную стоимость; прибыль организаций") \
        .startswith("НДС")
    assert kb_module.tax_topic("налог на доходы физических лиц").startswith("НДФЛ")
    assert kb_module.tax_topic("налог на прибыль организаций").startswith("налог на прибыль")
    assert kb_module.tax_topic("арендная плата") is None


@needs_data
def test_law_docs_get_tax_topic_in_index_and_label(kb):
    assert kb.documents["nk_164"].topic.startswith("НДС")
    assert kb.documents["nk_284"].topic.startswith("налог на прибыль")
    assert kb.documents["contract_a-08-2024"].topic is None
    assert "НДС" in kb.get_chunk("nk_164#p3").text_for_index
    assert "(НДС" in gen.doc_label(kb.documents["nk_164"])
    assert kb.documents["nk_164"].short_ref == "НК РФ, ст. 164"


@needs_data
@pytest.mark.parametrize("q", ["Какая общая ставка НДС, если льготные ставки не применяются?",
                               "Общая ставка НДС ведь 20%? Её и указываем в счёте?"])
def test_vat_question_finds_nk_164(kb, q):
    # Без реранкера (BM25) — статья о ставках НДС, а не налога на прибыль с тем же заголовком
    r = Retriever(kb, None)
    ids = [h.chunk.chunk_id for h in r.search(q, top_k=8)]
    assert any(i.startswith("nk_164#") for i in ids)
    assert not any(i.startswith(("nk_284#", "nk_224#")) for i in ids)
    if "льготные" not in q:
        return
    # Нужный пункт попадает в пул реранкера (для «ведь 20%?» BM25 без эмбеддингов его не находит)
    pool = []
    r._rerank = lambda query, p: pool.extend(p) or None
    r.search(q, top_k=8)
    assert "nk_164#p3" in [r.chunks[i].chunk_id for i in pool]


def test_prompt_answers_only_the_question():
    import prompts

    s = prompts.system_prompt(False)
    assert "Полнота" not in s and "обязательно коротко назови" not in s
    assert "Только по вопросу" in s


@needs_data
@pytest.mark.parametrize("status", ["insufficient", "offtopic", "refuse_illegal"])
def test_no_source_suffix_on_refusal(kb, status):
    reply = json.dumps({"status": status, "answer": "Сведений о МРОТ за 2024 год в базе нет.",
                        "citations": ["mrot_1#ch1"]}, ensure_ascii=False)
    chunk = kb.get_chunk("mrot_1#ch1")
    res = gen.generate(FakeClient(reply), kb, "МРОТ в 2024?", [], [chunk])
    assert "Источник" not in res["answer"]


@needs_data
def test_unknown_clause_ref_removed(kb):
    chunks = [kb.get_chunk("tk_99#ch6")]
    text = "Не более 120 часов в год (ТК РФ, ст. 99, ч. 6); можно до 240 часов по коллективному договору (п. 3.1)."
    out = gen.drop_unknown_clause_refs(text, chunks)
    assert "(п. 3.1)" not in out and "ст. 99, ч. 6" in out
    assert out.endswith("коллективному договору.")


@needs_data
@pytest.mark.parametrize("text", ["Приёмка по качеству — 5 рабочих дней (п. 6.1).",
                                  "Приёмка — 5 дней (п. 6.1, 9.9).",
                                  "Проверка дополняет порядок регламента (п. 5.4)."])
def test_known_clause_ref_kept(kb, text):
    chunks = [kb.get_chunk("contract_p-17-2024#p6.1"), kb.get_chunk("letter_k-12#whole")]
    assert gen.drop_unknown_clause_refs(text, chunks) == text


@needs_data
def test_generate_drops_invented_clause(kb):
    reply = json.dumps({"status": "answer", "answer": "Плата — 450 000 рублей (п. 3.1), залог (п. 7.7).",
                        "citations": ["contract_a-08-2024#p3.1"]}, ensure_ascii=False)
    res = gen.generate(FakeClient(reply), kb, "Аренда?", [], [kb.get_chunk("contract_a-08-2024#p3.1")])
    assert "(п. 3.1)" in res["answer"] and "7.7" not in res["answer"]


A08 = ["contract_a-08-2024#p3.1", "contract_a-08-2024#p3.2", "contract_a-08-2024#p3.4", "contract_a-08-2024#p3.5"]


@needs_data
def test_fallback_call_after_two_failures(kb, monkeypatch):
    import llm

    monkeypatch.delenv("LLM_NO_THINK", raising=False)
    client = FakeClient([llm.Reply("", "length"), llm.Reply("", "length"),
                         llm.Reply("Арендная плата — 450 000 рублей в месяц (п. 3.1).", "stop")])
    res = gen.generate(client, kb, "Какая аренда?", [], [kb.get_chunk(c) for c in A08])
    assert client.calls["chat"] == 3
    assert res["answer"].startswith("Арендная плата — 450 000 рублей") and res["status"] == "answer"
    assert res["fallback"] is True and res["raw_ok"] is False
    assert [c["chunk_id"] for c in res["citations"]] == A08[:1]  # цитируем только использованный пункт
    assert client.max_tokens[2] == gen.MAX_RETRY_TOKENS
    assert fragment_ids(client.messages[2]) == A08[:3]
    assert "JSON" not in client.messages[2][0]["content"] and "/no_think" not in client.messages[2][-1]["content"]


@needs_data
def test_fallback_no_think_flag(kb, monkeypatch):
    import llm

    monkeypatch.setenv("LLM_NO_THINK", "1")
    client = FakeClient([llm.Reply("", "length"), llm.Reply("", "length"), llm.Reply("Плата 450 000 (п. 3.1).")])
    gen.generate(client, kb, "Какая аренда?", [], [kb.get_chunk(A08[0])])
    assert client.messages[2][-1]["content"].rstrip().endswith("/no_think")


@needs_data
def test_fallback_failure_returns_stub(kb):
    import llm

    client = FakeClient(llm.Reply("", "length"))
    res = gen.generate(client, kb, "Какая аренда?", [], [kb.get_chunk(A08[0])])
    assert client.calls["chat"] == 3 and res["answer"] == gen.FAIL_ANSWER and res["status"] == "error"


@needs_data
def test_tax_filter_uses_current_question_first(kb):
    r = Retriever(kb, None)
    hits = r.search("а какая ставка налога на прибыль для иностранных организаций?", top_k=8,
                    context="Какая общая ставка НДС?", entity_context="Какая общая ставка НДС?")
    ids = [h.chunk.chunk_id for h in hits]
    assert any(i.startswith("nk_284#") for i in ids) and not any(i.startswith("nk_164#") for i in ids)


@needs_data
def test_tax_filter_ignores_assistant_only_mention(kb):
    from search import query_tax_topic
    assert query_tax_topic("ставка для нерезидентов") is None


@needs_data
def test_fallback_cites_only_used_fragments(kb):
    import llm

    client = FakeClient([llm.Reply("", "length"), llm.Reply("", "length"),
                         llm.Reply("Арендная плата — 450 000 рублей в месяц (п. 3.1).", "stop")])
    res = gen.generate(client, kb, "Какая аренда?", [], [kb.get_chunk(c) for c in A08])
    assert [c["chunk_id"] for c in res["citations"]] == [A08[0]]


@needs_data
def test_fallback_refusal_is_insufficient_without_source(kb):
    import llm

    client = FakeClient([llm.Reply("", "length"), llm.Reply("", "length"),
                         llm.Reply("Во фрагментах нет сведений о сроке оплаты; не хватает договора.", "stop")])
    res = gen.generate(client, kb, "Срок оплаты?", [], [kb.get_chunk(c) for c in A08])
    assert res["status"] == "insufficient" and "Источник" not in res["answer"] and res["citations"] == []


@needs_data
def test_fallback_rejects_thinking_text(kb):
    import llm

    client = FakeClient([llm.Reply("", "length"), llm.Reply("", "length"),
                         llm.Reply("<think>Okay, the user asks about rent. Let me check.", "length")])
    res = gen.generate(client, kb, "Какая аренда?", [], [kb.get_chunk(A08[0])])
    assert res["answer"] == gen.FAIL_ANSWER


@needs_data
def test_no_fallback_when_llm_went_down(kb):
    import llm

    client = FakeClient([llm.Reply("", "length"), RuntimeError("down"), llm.Reply("Плата (п. 3.1).")])
    res = gen.generate(client, kb, "Какая аренда?", [], [kb.get_chunk(A08[0])])
    assert client.calls["chat"] == 2 and res["answer"] == gen.FAIL_ANSWER


@needs_data
@pytest.mark.parametrize("text", ["Арендная плата (п. 3).", "Плата вносится (пп. 3.1)."])
def test_clause_prefix_of_known_ref_kept(kb, text):
    assert gen.drop_unknown_clause_refs(text, [kb.get_chunk(c) for c in A08[:2]]) == text


def test_known_ref_from_podpunkt_and_history():
    class C:
        ref, text = "п. 1 ст. 164", "в соответствии с подпунктом 4 настоящего пункта"
    assert gen.drop_unknown_clause_refs("Ставка 0% (пп. 4).", [C()]) == "Ставка 0% (пп. 4)."
    assert gen.drop_unknown_clause_refs("Как я писал (п. 9.9).", [C()], history="ранее (п. 9.9)") \
        == "Как я писал (п. 9.9)."


def test_fallback_prompt_refuses_illegal_and_offtopic():
    import prompts
    assert "противоправ" in prompts.FALLBACK_SYSTEM and "на «вы»" in prompts.FALLBACK_SYSTEM
