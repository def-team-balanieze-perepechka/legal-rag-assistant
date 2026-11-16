from __future__ import annotations

import json
import os
import re
import sys

import checks
import prompts

STATUSES = {"answer", "insufficient", "false_premise", "refuse_illegal", "offtopic", "clarify"}
NO_SOURCE_STATUSES = {"insufficient", "offtopic", "refuse_illegal"}
FAIL_ANSWER = ("Не удалось сформировать ответ по базе знаний."
               "Попробуйте переформулировать вопрос.")
MAX_FRAGMENT_CHARS = 2600
# Лимит ответа, если клиент не задал свой. Рассуждающие модели тратят его и на рассуждения:
# при обрыве по лимиту повторяем тот же запрос с удвоенным лимитом, но не выше MAX_RETRY_TOKENS.
DEFAULT_MAX_TOKENS = 4096
MAX_RETRY_TOKENS = 8192
FALLBACK_FRAGMENTS = 3


def doc_label(doc) -> str:
    label = doc.short_ref
    if getattr(doc, "topic", None):
        label += f" ({doc.topic})"
    if getattr(doc, "counterparty", None) and doc.doc_type in ("contract", "addendum"):
        label += f" с {doc.counterparty['name']}"
    if doc.status == "expired":
        until = f" с {doc.valid_to:%d.%m.%Y}" if doc.valid_to else ""
        label += f" (УТРАТИЛ СИЛУ{until})"
    elif doc.valid_from and doc.doc_type in ("regulation", "letter", "memo", "addendum"):
        label += f" (действует с {doc.valid_from:%d.%m.%Y})"
    return label


def edition_label(doc, event_date) -> str:
    """
    Метка редакции регламента относительно даты события;
    для остальных документов пусто.
    """
    if event_date is None or doc.doc_type != "regulation":
        return ""
    active = (doc.valid_from is None or doc.valid_from <= event_date) and (doc.valid_to is None or event_date < doc.valid_to)
    return " | действовала на дату события" if active else " | на дату события не действовала"


def format_fragments(chunks, kb, event_date=None) -> str:
    if not chunks:
        return prompts.NO_FRAGMENTS
    parts = []
    for c in chunks:
        doc = kb.documents[c.doc_id]
        head = f"[{c.chunk_id}] {doc_label(doc)} | {c.ref}{edition_label(doc, event_date)}"
        if c.section:
            head += f" | раздел «{c.section}»"
        notes = []
        if c.meta.get("amended_by"):
            adds = ", ".join(kb.documents[a].short_ref for a in c.meta["amended_by"] if a in kb.documents)
            notes.append(f"Этот пункт изменён: {adds} "
                         f"— см. условия применения в его фрагменте.")
        amends = getattr(doc, "amends", None)
        if amends and amends.get("condition"):
            notes.append(f"Условие применения изменений: {amends['condition']}.")
        if c.repealed:
            notes.append("Норма утратила силу.")
        if c.meta.get("suspicious"):
            notes.append(prompts.SUSPICIOUS_NOTE)
        text = c.text if len(c.text) <= MAX_FRAGMENT_CHARS else c.text[:MAX_FRAGMENT_CHARS] + " …"
        parts.append("\n".join([head, *notes, text]))
    return "\n\n".join(parts)


def format_history(history: list[dict] | None) -> str:
    if not history:
        return ""
    names = {"user": "Пользователь", "assistant": "Ассистент"}
    turns = "\n".join(f"{names.get(t['role'], t['role'])}: {t['content']}" for t in history)
    return prompts.HISTORY.format(turns=turns)


def parse_json(text: str) -> dict | None:
    """
    Достаtт первый JSON-объект из ответа модели
    (с ```-обtртками, текстом вокруг и т. п.).
    """
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


_FIELD_END_RE = re.compile(r'"\s*,\s*"(?:status|citations|missing_info)"\s*:|"\s*}\s*$')


_ABBR_RE = re.compile(r"(?:^|[\s(])(?:п|пп|ч|ст|абз|прим|г|гг|руб|т|д|см|ред|им|ул|№)$", re.I)


def trim_to_sentence(text: str) -> str:
    """
    Обрезает оборванный текст до последнего законченного
    предложения (сокращения «п.», «ст.» — не конец).
    """
    for m in reversed(list(re.finditer(r"[.!?](?=\s|$)", text))):
        if not _ABBR_RE.search(text[:m.start()]):
            return text[:m.end()].strip()
    return ""


def salvage_json(text: str, cut: bool = False) -> dict | None:
    """
    Достает поля из сломанного JSON: неэкранированные кавычки внутри answer,
    ответ, оборванный по лимиту токенов (cut=True — обрезаем до последнего
    законченного предложения). Лучше частичный ответ, чем заглушка.
    """
    if not text:
        return None
    m = re.search(r'"answer"\s*:\s*"', text)
    if not m:
        return None
    rest = text[m.end():]
    end = _FIELD_END_RE.search(rest)
    answer = rest[:end.start()] if end else rest.rstrip().rstrip('"}` \n')
    answer = answer.replace('\\"', '"').replace("\\n", " ").strip()
    if cut and not end:
        answer = trim_to_sentence(answer)
    if not answer:
        return None
    status = re.search(r'"status"\s*:\s*"(\w+)"', text)
    cites = re.search(r'"citations"\s*:\s*\[(.*?)\]', text, re.S)
    return {
        "status": status.group(1) if status else "answer",
        "answer": answer,
        "citations": re.findall(r'"([^"]+)"', cites.group(1)) if cites else [],
        "missing_info": "",
    }


def one_line(text: str) -> str:
    return " ".join(str(text).split())


# Ссылка на пункт или статью уже в тексте;
# номер договора («№ А-08/2024») ссылкой на пункт не считается
_REF_HINT_RE = re.compile(r"\b(п|пп|ч|ст|абз|прим)\.\s*\d|раздел|преамбул|записк|письм", re.I)


_NUM = r"\d+(?:\.\d+)*"
# «(п. 3.1)», «(п. 3.1, 3.2)», «(пп. 1 и 2)», «(ч. 6)» — скобка только с номерами пунктов
_CLAUSE_PAREN_RE = re.compile(rf"\s*\((?:пп?|ч)\.\s*({_NUM}(?:\s*(?:,|и)\s*{_NUM})*)\)", re.I)
# Номера пунктов, известные по фрагментам: в ref («п. 3 ст. 164»),
# в тексте («п. 5.4 Регламента») и в начале нумерованных абзацев («6.1. Приёмка…»)
_KNOWN_REF_RE = re.compile(rf"(?:\b(?:пп?|ч)|пункт\w*|част\w*)\.?\s*({_NUM})", re.I)  # и «подпункт»
_LEADING_NUM_RE = re.compile(rf"(?m)^\s*({_NUM})[.)]\s")


def drop_unknown_clause_refs(answer: str, chunks, history: str = "") -> str:
    """
    Убирает «(п. N)», если ни одного из номеров нет во фрагментах,
    переданных модели, и в истории.
    """
    known = set(_KNOWN_REF_RE.findall(history or ""))
    for c in chunks:
        for src in (c.ref or "", c.text or ""):
            known.update(_KNOWN_REF_RE.findall(src))
        known.update(_LEADING_NUM_RE.findall(c.text or ""))

    def repl(m: re.Match) -> str:
        nums = re.findall(_NUM, m.group(1))
        # «п. 3» — раздел, если известен «п. 3.1»
        if any(n in known or any(k.startswith(n + ".") for k in known) for n in nums):
            return m.group(0)
        print(f"[generate] убрана ссылка на пункт, которого нет во фрагментах: {m.group(0).strip()}",
              file=sys.stderr)
        return ""

    return _CLAUSE_PAREN_RE.sub(repl, answer)


def build_citations(ids: list[str], kb) -> list[dict]:
    out = []
    for cid in ids:
        c = kb.get_chunk(cid)
        doc = kb.documents[c.doc_id]
        out.append({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "title": doc.title,
            "short_ref": doc.short_ref,
            "ref": c.ref,
            "quote": c.text[:400],
            "start": c.start,
            "end": c.end,
            "edition": doc.edition,
            "status": doc.status,
        })
    return out


def source_suffix(citations: list[dict]) -> str:
    """«Источник: договор аренды № А-08/2024, п. 3.1; …» — собирает код, а не модель."""
    seen, parts = set(), []
    for c in citations[:3]:
        key = (c["short_ref"], c["ref"])
        if key in seen:
            continue
        seen.add(key)
        parts.append(c["short_ref"] if c["ref"] == c["short_ref"] else f"{c['short_ref']}, {c['ref']}")
    return "Источник: " + "; ".join(parts) + "." if parts else ""


def fallback_answer(client, kb, question: str, history: list[dict] | None, chunks: list) -> dict | None:
    """
    Третья попытка вместо заглушки.
    Короткий промпт без JSON, лучшие фрагменты, максимальный лимит.
    """
    user = prompts.FALLBACK_USER.format(question=question, fragments=format_fragments(chunks, kb))
    if os.getenv("LLM_NO_THINK") == "1":  # qwen3 в Ollama: без рассуждений, чтобы не съесть лимит
        user += prompts.NO_THINK
    messages = [{"role": "system", "content": prompts.FALLBACK_SYSTEM},
                {"role": "user", "content": format_history(history) + user}]
    try:
        raw = client.chat(messages, max_tokens=MAX_RETRY_TOKENS)
    except Exception as exc:
        print(f"[generate] запасной вызов не удался: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    text = one_line(raw)
    if "<think" in text.lower():  # Оборвался внутри рассуждений — это не ответ
        return None
    if getattr(raw, "finish_reason", None) == "length":
        text = trim_to_sentence(text)
    if not text or text.startswith("{"):
        return None
    print("[generate] ответ из запасного вызова", file=sys.stderr)
    answer = drop_unknown_clause_refs(text, chunks, _history_text(history))
    if _FALLBACK_REFUSAL_RE.search(answer):
        # Отказ без подписи: перечисленные пункты судья считает фактами
        return {"status": "insufficient", "answer": answer, "citations": [], "missing_info": None,
                "dropped_citations": [], "raw_ok": False, "fallback": True}
    nums = set(_KNOWN_REF_RE.findall(answer))
    used = [c for c in chunks if nums & set(_KNOWN_REF_RE.findall(c.ref or ""))] or chunks[:1]
    citations = build_citations([c.chunk_id for c in used], kb)
    if not _REF_HINT_RE.search(answer):
        answer = f"{answer} {source_suffix(citations)}".strip()
    return {"status": "answer", "answer": answer, "citations": citations, "missing_info": None,
            "dropped_citations": [], "raw_ok": False, "fallback": True}


_FALLBACK_REFUSAL_RE = re.compile(
    r"нет\s+(?:сведений|информации|данных)|не\s+содерж|не\s+хватает|отсутству|не\s+удалось\s+найти", re.I)


def _history_text(history: list[dict] | None) -> str:
    return " ".join(t.get("content", "") for t in history or [])


def plan_extra(plan, kb, chunks=()) -> str:
    """
    Блок «ПЛАН» и строка об отсутствующих статьях после фрагментов;
    без плана пусто (как в v2a).
    """
    if plan is None:
        return ""
    refs = [kb.documents[d].short_ref for d in plan.docs if d in kb.documents]
    hint = prompts.plan_hint(plan, refs)
    parts = ([f"ПЛАН: {hint}"] if hint else [])
    note = checks.missing_article_note(chunks, kb)
    if note:
        parts.append(note)
    return "\n\n" + "\n".join(parts) if parts else ""


def generate(client, kb, question: str, history: list[dict] | None, chunks: list,
             allow_clarify: bool = False, plan=None) -> dict:
    messages = [
        {"role": "system", "content": prompts.system_prompt(allow_clarify)},
        {"role": "user", "content": prompts.USER.format(
            history=format_history(history), question=question,
            fragments=format_fragments(chunks, kb, plan.event_date if plan else None),
            extra=plan_extra(plan, kb, chunks))},
    ]
    raw, obj, cut, llm_ok, llm_down = "", None, False, False, False
    limit = getattr(getattr(client, "s", None), "max_tokens", None) or DEFAULT_MAX_TOKENS
    for attempt in range(2):
        try:
            raw = client.chat(messages, max_tokens=limit)
        except Exception as exc:
            print(f"[generate] LLM недоступна: {type(exc).__name__}: {exc}", file=sys.stderr)
            llm_down = True
            break
        llm_ok = True
        cut = getattr(raw, "finish_reason", None) == "length"
        # оборванный ответ не чиним сразу: сначала повтор с бОльшим лимитом
        obj = parse_json(raw) or (None if cut else salvage_json(raw))
        if obj and str(obj.get("answer", "")).strip():
            break
        obj = None
        if cut:
            limit = min(MAX_RETRY_TOKENS, limit * 2)
        else:
            messages = messages + [{"role": "assistant", "content": raw},
                                   {"role": "user", "content": prompts.RETRY_JSON}]
    if obj is None and raw:
        obj = salvage_json(raw, cut=cut)

    if obj is None:
        text = one_line(raw)
        if text and not text.startswith("{"):
            # модель ответила текстом без JSON — отдаём как есть, но без цитат
            return {"status": "answer", "answer": text, "citations": [], "missing_info": None,
                    "dropped_citations": [], "raw_ok": False}
        if llm_ok and not llm_down and chunks:
            fallback = fallback_answer(client, kb, question, history, chunks[:FALLBACK_FRAGMENTS])
            if fallback:
                return fallback
        return {"status": "error", "answer": FAIL_ANSWER, "citations": [], "missing_info": None,
                "dropped_citations": [], "raw_ok": False}

    allowed = {c.chunk_id for c in chunks}
    cited = obj.get("citations") or []
    if not isinstance(cited, list):
        cited = [cited]
    valid, dropped = [], []
    for cid in cited:
        cid = str(cid).strip().strip("[]")
        if cid in allowed and cid not in valid:
            valid.append(cid)
        elif cid not in valid:
            dropped.append(cid)
    if dropped:
        print(f"[generate] отброшены неизвестные chunk_id: {dropped}", file=sys.stderr)

    status = str(obj.get("status", "answer")).strip().lower()
    if status not in STATUSES:
        status = "answer"
    if status == "clarify" and not allow_clarify:
        status = "answer"

    citations = build_citations(valid, kb)
    answer = drop_unknown_clause_refs(one_line(obj.get("answer", "")), chunks, _history_text(history))
    if plan is not None:
        answer = checks.drop_invented_articles(answer, status, chunks, kb)
        answer = checks.drop_leaked_chunk_ids(answer, kb)
    # К отказу подпись не нужна: судья считает перечисленные пункты фактами, которых нет в эталоне
    if citations and status not in NO_SOURCE_STATUSES and not _REF_HINT_RE.search(answer):
        answer = f"{answer} {source_suffix(citations)}".strip()
    missing = one_line(obj.get("missing_info") or "") or None
    return {"status": status, "answer": answer, "citations": citations, "missing_info": missing,
            "dropped_citations": dropped, "raw_ok": True}
