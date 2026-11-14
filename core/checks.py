from __future__ import annotations

import re
import sys

import generate as gen
import prompts

NO_SOURCE = {"insufficient", "offtopic", "refuse_illegal"}
OTHER_ACT_WINDOW = 40

_NUM = r"\d+(?:\.\d+)?"
_NUM_RE = re.compile(_NUM)
# «статьи 6», «статьей 10», «статей 6, 10 и 11», «ст. 260»
_ART_RE = re.compile(
    rf"(?:\bстат(?:ья|ьи|ье|ьей|ьёй|ью|ей|ьям|ьями|ьях)|\bст\.)\s*({_NUM}(?:\s*(?:,|и)\s*{_NUM})*)", re.I)
_OTHER_ACT_RE = re.compile(r"кодекс|закон|коап|фз", re.I)
_OWN_ACT_RE = re.compile(r"настоящ\w+\s+(?:федеральн\w+\s+)?(?:закон|кодекс)\w*", re.I)
_PAREN_RE = re.compile(r"\s*\([^()]*\)")


def _num_key(n: str) -> tuple:
    return tuple(int(x) for x in n.split("."))


def _refs(text: str):
    """(номер статьи, текст после номера) для каждой отсылки на статью."""
    for m in _ART_RE.finditer(text):
        base = m.start(1)
        for n in _NUM_RE.finditer(m.group(1)):
            end = base + n.end()
            yield n.group(0), text[end:end + OTHER_ACT_WINDOW]


def _kb_articles(kb, law_code: str | None = None) -> set[str]:
    return {d.article for d in kb.documents.values()
            if d.article and d.doc_type == "law" and (law_code is None or d.law_code == law_code)}


def missing_article_note(chunks, kb) -> str:
    """
    Статьи закона, на которые ссылаются фрагменты,
    но которых нет в базе (до генерации).
    """
    missing: dict[str, set[str]] = {}
    for c in chunks:
        doc = kb.documents.get(c.doc_id)
        if doc is None or doc.doc_type != "law" or not doc.law_code:
            continue
        known = _kb_articles(kb, doc.law_code)
        for num, tail in _refs(c.text or ""):
            if _OTHER_ACT_RE.search(_OWN_ACT_RE.sub("", tail)):
                continue
            if num not in known:
                missing.setdefault(doc.law_code, set()).add(num)
    parts = []
    for code, nums in missing.items():
        listed = ", ".join(sorted(nums, key=_num_key))
        parts.append(f"Статей {listed} {code} в базе нет: не делай вывода по ним, скажи, что их нет.")
    return " ".join(parts)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    for p in parts:
        if out and gen._ABBR_RE.search(out[-1][:-1]):
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def drop_invented_articles(answer: str, status: str, chunks, kb) -> str:
    """
    В отказах убирает предложения со ссылкой на статью, которой нет ни в базе,
    ни во фрагментах.
    """
    if status not in NO_SOURCE:
        return answer
    known = _kb_articles(kb)
    for c in chunks:
        known.update(n for n, _ in _refs(c.text or ""))
    sentences = _split_sentences(answer)
    if not any(s.strip() for s in sentences):
        return answer
    kept = [s for s in sentences if all(n in known for n, _ in _refs(s))]
    if not kept:  # Выдумана каждая ссылка: оставлять выдумку нельзя, говорим нейтрально
        print("[checks] все предложения со ссылками на отсутствующие статьи, "
              "ответ заменен нейтральным", file=sys.stderr)
        return prompts.NEUTRAL_REFUSAL[status]
    if len(kept) == len(sentences):
        return answer
    print(f"[checks] убраны предложения с выдуманными статьями: {len(sentences) - len(kept)}", file=sys.stderr)
    return " ".join(kept)


def drop_leaked_chunk_ids(answer: str, kb) -> str:
    """Убирает скобки, внутри которых chunk_id из базы."""
    def repl(m: re.Match) -> str:
        inner = m.group(0)
        if any(cid in inner for cid in re.findall(r"[^\s()\[\],;]+#[^\s()\[\],;]+", inner)
               if cid.strip(".") in kb._chunk_by_id):
            return ""
        return inner
    return _PAREN_RE.sub(repl, answer)
