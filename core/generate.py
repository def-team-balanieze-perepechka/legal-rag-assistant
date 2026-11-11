from __future__ import annotations

import json
import re
import sys

import prompts

STATUSES = {"answer", "insufficient", "false_premise", "refuse_illegal", "offtopic", "clarify"}
FAIL_ANSWER = ("Не удалось сформировать ответ по базе знаний."
               "Попробуйте переформулировать вопрос.")
MAX_FRAGMENT_CHARS = 2600
# Лимит ответа, если клиент не задал свой. Рассуждающие модели тратят его и на рассуждения:
# при обрыве по лимиту повторяем тот же запрос с удвоенным лимитом, но не выше MAX_RETRY_TOKENS.
DEFAULT_MAX_TOKENS = 4096
MAX_RETRY_TOKENS = 8192


def doc_label(doc) -> str:
    label = doc.short_ref
    if getattr(doc, "counterparty", None) and doc.doc_type in ("contract", "addendum"):
        label += f" с {doc.counterparty['name']}"
    if doc.status == "expired":
        until = f" с {doc.valid_to:%d.%m.%Y}" if doc.valid_to else ""
        label += f" (УТРАТИЛ СИЛУ{until})"
    elif doc.valid_from and doc.doc_type in ("regulation", "letter", "memo", "addendum"):
        label += f" (действует с {doc.valid_from:%d.%m.%Y})"
    return label


def format_fragments(chunks, kb) -> str:
    if not chunks:
        return prompts.NO_FRAGMENTS
    parts = []
    for c in chunks:
        doc = kb.documents[c.doc_id]
        head = f"[{c.chunk_id}] {doc_label(doc)} | {c.ref}"
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


def generate(client, kb, question: str, history: list[dict] | None, chunks: list,
             allow_clarify: bool = False) -> dict:
    messages = [
        {"role": "system", "content": prompts.system_prompt(allow_clarify)},
        {"role": "user", "content": prompts.USER.format(
            history=format_history(history), question=question, fragments=format_fragments(chunks, kb))},
    ]
    raw, obj, cut = "", None, False
    limit = getattr(getattr(client, "s", None), "max_tokens", None) or DEFAULT_MAX_TOKENS
    for attempt in range(2):
        try:
            raw = client.chat(messages, max_tokens=limit)
        except Exception as exc:
            print(f"[generate] LLM недоступна: {type(exc).__name__}: {exc}", file=sys.stderr)
            break
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
    answer = one_line(obj.get("answer", ""))
    if citations and not _REF_HINT_RE.search(answer):
        answer = f"{answer} {source_suffix(citations)}".strip()
    missing = one_line(obj.get("missing_info") or "") or None
    return {"status": status, "answer": answer, "citations": citations, "missing_info": missing,
            "dropped_citations": dropped, "raw_ok": True}
