import streamlit as st

from ui_docview import open_document


def _close_panel() -> None:
    st.session_state.selected_msg = None


def render_reference_panel(panel_col) -> None:
    """Правая панель: цитаты того ответа, у которого нажали «📎 Источники»."""
    if panel_col is None:
        return
    idx = st.session_state.selected_msg
    messages = st.session_state.messages
    if idx is None or idx >= len(messages):
        return

    citations = messages[idx].get("citations") or []
    with panel_col:
        st.button("✕", key="close_panel_btn", on_click=_close_panel)
        st.subheader("📚 Источники")

        for i, c in enumerate(citations, 1):
            with st.container(border=True):
                st.markdown(f"**{i}. {c['short_ref']}**")
                place = c["ref"] if c["ref"] != c["short_ref"] else ""
                edition = f"редакция: {c['edition']}" if c.get("edition") else ""
                meta = " · ".join(x for x in (place, edition) if x)
                if meta:
                    st.caption(meta)
                if c.get("title") and c["title"] != c["short_ref"]:
                    st.caption(c["title"])
                if c.get("status") == "expired":
                    st.warning("Срок действия документа истёк")
                fragment = (c.get("text") or c["quote"]).strip().replace("\n", "\n> ")
                st.markdown(f"> {fragment}")
                st.button(
                    "📄 Открыть в документе",
                    key=f"open_{idx}_{i}",
                    on_click=open_document,
                    args=(c["doc_id"], c.get("start"), c.get("end")),
                )
