from __future__ import annotations

import re

import generate as gen
from judge_cases import NO_ANSWER_SCENARIOS
from search import number_key

BASELINE_FALLBACK = "Не удалось получить ответ."

_DOC_NUMBER_RE = re.compile(r"(?<![\w-])[A-Za-zА-Яа-яЁё]{1,3}-\d+(?:/\d+)?")
# «п. 2 ст. 9», «ч. 5 ст. 70» — ссылки внутри закона: их пункт не сверяем с документами компании
_LAW_CLAUSE_RE = re.compile(r"\b(?:пп|п|ч)\.\s*\d+(?:\.\d+)*\s*,?\s*(?=ст\.)", re.I)
# «ст. 164, п. 3» — тот же случай с обратным порядком
_LAW_CLAUSE_AFTER_RE = re.compile(r"\bст\.\s*\d+(?:\.\d+)?(?:\s*,?\s*(?:пп|п|ч)\.\s*\d+(?:\.\d+)*)+", re.I)
# Документы компании, названные словом, а не номером: берём все документы этого типа
_TYPE_WORDS = {"regulation": re.compile(r"регламент", re.I), "addendum": re.compile(r"соглашени", re.I),
               "letter": re.compile(r"письм", re.I), "memo": re.compile(r"записк", re.I)}
_CLAUSE_RE = re.compile(r"\bп\.\s*(\d+(?:\.\d+)*)", re.I)
_ARTICLE_RE = re.compile(r"\bст\.\s*(\d+(?:\.\d+)?)", re.I)
_ABSENT_RE = re.compile(r"нет\b|отсутств|не представлен|не содерж|не входит", re.I)
_NEAR = 80  # Символов вокруг ссылки на статью, где ищем оговорку «нет в базе»


def _company_docs_by_number(kb) -> dict[str, str]:
    return {number_key(d.number): d.doc_id for d in kb.documents.values()
            if d.number and d.doc_type != "law" and len(number_key(d.number)) >= 3}


def _named_docs(answer: str, kb) -> set[str]:
    by_number = _company_docs_by_number(kb)
    return {by_number[k] for k in (number_key(m) for m in _DOC_NUMBER_RE.findall(answer)) if k in by_number}


def _clauses(kb, doc_id: str) -> set[str]:
    return {m for c in kb.chunks if c.doc_id == doc_id for m in _CLAUSE_RE.findall(c.ref)}


def _related(kb, doc_id: str) -> set[str]:
    doc = kb.documents[doc_id]
    out = {doc_id, *doc.addenda}
    if doc.amends and doc.amends.get("doc_id"):
        out.add(doc.amends["doc_id"])
    return out


def _by_type_words(answer: str, kb) -> set[str]:
    types = {t for t, rx in _TYPE_WORDS.items() if rx.search(answer)}
    return {d.doc_id for d in kb.documents.values() if d.doc_type in types}


def _bad_ref(answer: str, kb, named: set[str]) -> bool:
    company = {d for d in named | _by_type_words(answer, kb) if kb.documents[d].doc_type != "law"}
    clause_sets = [_clauses(kb, d) for d in company]
    # Если назван документ без нумерованных пунктов (допсоглашение, письмо, записка),
    # пункт может относиться к нему - сверить нельзя, не проверяем
    if company and all(clause_sets):
        known = set().union(*clause_sets)
        text = _LAW_CLAUSE_AFTER_RE.sub(" ", _LAW_CLAUSE_RE.sub(" ", answer))
        if any(c not in known for c in _CLAUSE_RE.findall(text)):
            return True
    articles = {d.article for d in kb.documents.values() if d.doc_type == "law" and d.article}
    for m in _ARTICLE_RE.finditer(answer):
        if m.group(1) in articles:
            continue
        # «ст. 6 в базе нет» — честная оговорка рядом со ссылкой, не выдумка
        if not _ABSENT_RE.search(answer[max(0, m.start() - _NEAR):m.end() + _NEAR]):
            return True
    return False


def check_answer(answer: str, kb, case, status: str | None = None, citations: list[dict] | None = None,
                 raw_ok: bool | None = None) -> list[str]:
    flags = []
    text = (answer or "").strip()
    if not text or text in (gen.FAIL_ANSWER, BASELINE_FALLBACK) or status in ("error", "crash"):
        flags.append("stub")
    if raw_ok is False:
        flags.append("raw_text")
    by_number = _named_docs(text, kb)
    cited = {c["doc_id"] for c in citations or [] if c.get("doc_id") in kb.documents}
    if _bad_ref(text, kb, by_number | cited):
        flags.append("bad_ref")
    if citations is not None:
        allowed = set().union(*(_related(kb, d) for d in cited)) if cited else set()
        if by_number - allowed:
            flags.append("foreign_doc")
    if status in ("insufficient", "offtopic") and case.scenario not in NO_ANSWER_SCENARIOS:
        flags.append("false_refusal")
    words = len(text.split())
    if words > 150:
        flags.append("too_long")
    elif words < 15:
        flags.append("too_short")
    return flags
