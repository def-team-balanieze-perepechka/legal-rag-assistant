"""Просмотр документа базы знаний (ТЗ п.06): полный текст + подсветка цитируемого фрагмента."""
from __future__ import annotations

import html

import streamlit as st

from ui_api import load_document

CONTEXT_CHARS = 700   # сколько текста показывать вокруг фрагмента в «коротком» режиме


def open_document(doc_id: str, start: int | None = None, end: int | None = None) -> None:
    """on_click-колбэк: только запоминает запрос; сам диалог открывает app.py (см. show_pending_document)."""
    st.session_state.open_doc = {"doc_id": doc_id, "start": start, "end": end}


def _to_html(text: str, start: int | None, end: int | None) -> str:
    esc = html.escape
    if start is None or end is None or not (0 <= start < end <= len(text)):
        body = esc(text)
    else:
        body = f"{esc(text[:start])}<mark>{esc(text[start:end])}</mark>{esc(text[end:])}"
    return (
        '<div style="white-space:pre-wrap;line-height:1.55;font-size:0.95rem">'
        f"{body}</div>"
    )


@st.dialog("Документ", width="large")
def _dialog(doc_id: str, start: int | None, end: int | None) -> None:
    doc = load_document(doc_id)
    if doc is None:
        st.error("Не удалось загрузить документ. Проверьте, что API доступен.")
        return

    st.subheader(doc["short_ref"])
    meta = [doc["title"] if doc["title"] != doc["short_ref"] else "",
            f"редакция: {doc['edition']}" if doc.get("edition") else "",
            f"действует с {doc['valid_from']}" if doc.get("valid_from") else "",
            f"до {doc['valid_to']}" if doc.get("valid_to") else ""]
    meta = " · ".join(x for x in meta if x)
    if meta:
        st.caption(meta)
    if doc["status"] == "expired":
        st.warning("Срок действия документа истёк")

    text = doc["text"]
    has_fragment = start is not None and end is not None
    show_all = True
    if has_fragment:
        show_all = st.toggle("Показать документ целиком", value=False)

    if has_fragment and not show_all:
        lo = max(0, start - CONTEXT_CHARS)
        hi = min(len(text), end + CONTEXT_CHARS)
        st.html(_to_html(text[lo:hi], start - lo, end - lo))
        st.caption(f"Показан фрагмент с контекстом ({lo + 1}–{hi} из {len(text)} символов).")
    else:
        st.html(_to_html(text, start, end))
    if doc.get("source_url"):
        st.link_button("Источник документа", doc["source_url"])


def show_pending_document() -> None:
    """Вызывать в app.py на каждом прогоне: pop, чтобы закрытый крестиком диалог не открывался снова."""
    req = st.session_state.pop("open_doc", None)
    if req:
        _dialog(req["doc_id"], req["start"], req["end"])
