import streamlit as st

from ui_api import load_documents
from ui_docview import open_document

DOC_TYPE_TITLES = {
    "contract": "Договоры",
    "addendum": "Допсоглашения",
    "law": "Законы",
    "regulation": "Регламенты",
    "memo": "Служебные записки",
    "letter": "Письма",
}


def _reset_dialog() -> None:
    st.session_state.messages = []
    st.session_state.pending = False
    st.session_state.last_error = None
    st.session_state.selected_msg = None


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Юридический ассистент")
        st.button(
            "🗑 Новый диалог",
            width="stretch",
            on_click=_reset_dialog,
            disabled=not st.session_state.messages,
            help="Очищает историю: она уходит в API как контекст и влияет на поиск",
        )
        st.toggle(
            "Уточняющие вопросы",
            key="allow_clarify",
            help="Если из вопроса неясно, о каком договоре/периоде речь, ассистент уточнит перед ответом",
        )
        st.divider()

        docs = load_documents()
        st.subheader(f"База знаний ({len(docs)})")
        if not docs:
            st.caption("Список документов недоступен.")
            return

        groups: dict[str, list[dict]] = {}
        for d in docs:
            groups.setdefault(d["doc_type"], []).append(d)
        for doc_type, items in groups.items():
            with st.expander(f"{DOC_TYPE_TITLES.get(doc_type, 'Прочее')} ({len(items)})"):
                for d in items:
                    mark = " ⚠️" if d.get("status") == "expired" else ""
                    st.button(
                        f"{d['short_ref']}{mark}",
                        key=f"doc_{d['doc_id']}",
                        on_click=open_document,
                        args=(d["doc_id"],),
                        type="tertiary",
                        width="stretch",
                    )
        if any(d.get("status") == "expired" for d in docs):
            st.caption("⚠️ — срок действия документа истёк. Нажмите на документ, чтобы открыть его.")
