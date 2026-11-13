from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date

import generate as gen
import kb as kb_module
import prompts

SCENARIOS = ("fact", "false_premise", "date_edition", "multi_doc",
             "not_in_kb", "illegal", "offtopic", "old_law")
MAX_DOCS = 4
MAX_QUERIES = 3
MAX_QUERY_CHARS = 200
MAX_CLAIMS = 3
ROUTER_MAX_TOKENS = 512


@dataclass
class Plan:
    scenario: str = "fact"
    docs: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    event_date: date | None = None
    claims: list[str] = field(default_factory=list)

    def as_debug(self) -> dict:
        return {"scenario": self.scenario, "docs": self.docs, "queries": self.queries,
                "event_date": self.event_date.isoformat() if self.event_date else None,
                "claims": self.claims}


def router_enabled() -> bool:
    """
    Роутер включен по умолчанию;
    ROUTER=0 (false/no/off) возвращает путь v2a с одним вызовом на вопрос.
    """
    return os.getenv("ROUTER", "").strip().lower() not in ("0", "false", "no", "off")


def _status(doc) -> str:
    if doc.status == "expired":
        s = "утратил силу"
        if doc.valid_to:
            s += f" с {doc.valid_to:%d.%m.%Y}"
        return s
    if doc.valid_from:
        return f"действует с {doc.valid_from:%d.%m.%Y}"
    return "действует"


def build_catalog(kb) -> str:
    """Строка на документ: doc_id | short_ref | статус, даты | тема."""
    return "\n".join(
        f"{d.doc_id} | {d.short_ref} | {_status(d)} | {d.topic or d.title}"
        for d in kb.documents.values())


def _str_list(value, limit: int, max_len: int | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            continue
        item = item.strip()
        out.append(item[:max_len] if max_len else item)
        if len(out) >= limit:
            break
    return out


def _parse_date(value) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def parse_plan(raw: str, doc_ids: set[str]) -> Plan | None:
    obj = gen.parse_json(raw)
    if obj is None:
        return None
    scenario = obj.get("scenario")
    docs = [d for d in _str_list(obj.get("docs"), 10**6) if d in doc_ids]
    return Plan(
        scenario=scenario if scenario in SCENARIOS else "fact",
        docs=list(dict.fromkeys(docs))[:MAX_DOCS],
        queries=_str_list(obj.get("queries"), MAX_QUERIES, MAX_QUERY_CHARS),
        event_date=_parse_date(obj.get("event_date")),
        claims=_str_list(obj.get("claims"), MAX_CLAIMS),
    )


def route(client, kb, question: str, history: list[dict] | None) -> Plan | None:
    messages = [
        {"role": "system", "content": prompts.ROUTER_SYSTEM},
        {"role": "user", "content": prompts.ROUTER_USER.format(
            catalog=build_catalog(kb), history=gen.format_history(history), question=question)},
    ]
    try:
        raw = client.chat(messages, max_tokens=ROUTER_MAX_TOKENS)
    except Exception as exc:
        print(f"[router] вызов не удался: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    plan = parse_plan(raw or "", set(kb.documents))
    if plan is None:
        print("[router] ответ не разобран, идём путём v2a", file=sys.stderr)
        return None
    return adjust_plan(plan, kb, question)


_ORG_RE = re.compile(r"(?:ООО|АО|ПАО|ЗАО|ИП)\s*[«\"]([^»\"]+)[»\"]")
_CONTENT_SCENARIOS = ("fact", "false_premise", "multi_doc", "date_edition")


def names_unknown_company(question: str, kb) -> bool:
    """В вопросе названа организация, которой нет среди контрагентов базы."""
    others = [m.group(0) for m in _ORG_RE.finditer(question or "")
              if kb_module.SELF_ORG_NAME not in m.group(1).lower()]
    if not others:
        return False
    return not kb.match_counterparties(question) and not kb.resolve_contract(question)


def adjust_plan(plan: Plan, kb, question: str = "") -> Plan:
    """
    Правила поверх ответа модели: утверждения пользователя означают проверку предпосылки,
    а прошлое значение закона — не «редакция на дату», потому что законы в базе только действующие;
    названная организация, которой нет в базе, — это вопрос о том, чего в базе нет.
    """
    if plan.scenario in _CONTENT_SCENARIOS and names_unknown_company(question, kb):
        plan.scenario = "not_in_kb"
        plan.docs = []
    elif plan.scenario == "fact" and plan.claims:
        plan.scenario = "false_premise"
    elif plan.scenario == "date_edition" and plan.docs and all(
            kb.documents[d].doc_type == "law" for d in plan.docs):
        plan.scenario = "old_law"
    elif plan.scenario == "fact" and len({kb.documents[d].doc_type for d in plan.docs}) >= 2:
        plan.scenario = "multi_doc"
    return plan
