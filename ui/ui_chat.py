"""
Диалог с ассистентом: история из session_state.messages, вызов API, рендер ответов.

Формат сообщения в session_state.messages:
  user:      {"role": "user", "content": str}
  assistant: {"role": "assistant", "content": str, "status": str,
              "citations": list[dict], "missing_info": str | None, "debug": dict | None}
"""
import streamlit as st

from api.client import LegalAssistantError
from ui_api import DEBUG, ask, describe_error

GREETING = (
    "Здравствуйте! Я отвечаю на вопросы по документам базы знаний компании "
    "и ссылаюсь на конкретные пункты. Задайте вопрос."
)

# status -> (тип плашки Streamlit, текст)
STATUS_NOTES = {
    "false_premise": ("warning", "⚠️ В вопросе была неточность — ассистент её поправил."),
    "refuse_illegal": ("warning", "🚫 Запрос отклонён: ассистент не помогает с противоправными действиями."),
    "offtopic": ("info", "ℹ️ Вопрос вне базы знаний компании."),
    "clarify": ("info", "❓ Нужно уточнение — ответьте в чате."),
}


def _open_sources(idx: int) -> None:
    st.session_state.selected_msg = idx


def _retry() -> None:
    st.session_state.last_error = None
    st.session_state.pending = True


def _render_message(idx: int, msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("\n", "  \n"))
        if msg["role"] != "assistant":
            return

        status = msg.get("status", "answer")
        if status == "insufficient":
            st.info(f"**Чего не хватает в базе:** {msg.get('missing_info') or '—'}")
        elif status in STATUS_NOTES:
            kind, text = STATUS_NOTES[status]
            getattr(st, kind)(text)

        citations = msg.get("citations") or []
        if citations:
            st.button(
                f"📎 Источники ({len(citations)})",
                key=f"src_{idx}",
                on_click=_open_sources,
                args=(idx,),
            )
        if DEBUG and msg.get("debug"):
            with st.expander("Debug"):
                st.json(msg["debug"])


def _answer_pending() -> None:
    """Отправляет историю в API; пока ждём — показывает спиннер вместо ответа."""
    slot = st.empty()
    with slot.container():
        with st.chat_message("assistant"):
            with st.spinner("Ищу в базе знаний… (может занять до минуты)"):
                try:
                    data = ask(st.session_state.messages)
                except LegalAssistantError as exc:
                    text, retryable = describe_error(exc)
                    st.session_state.last_error = {"text": text, "retryable": retryable}
                    data = None
    slot.empty()
    st.session_state.pending = False

    if data is not None:
        st.session_state.messages.append({
            "role": "assistant",
            "content": data["answer"],
            "status": data.get("status", "answer"),
            "citations": data.get("citations") or [],
            "missing_info": data.get("missing_info"),
            "debug": data.get("debug"),
        })
        _render_message(len(st.session_state.messages) - 1, st.session_state.messages[-1])


def _render_error() -> None:
    err = st.session_state.last_error
    with st.chat_message("assistant"):
        st.error(err["text"])
        if err["retryable"]:
            st.button("🔄 Повторить", key="retry_btn", on_click=_retry)


def render_chat_content(chat_col) -> None:
    with chat_col:
        st.subheader("💬 Вопрос по базе знаний")

        # контейнер создаём ДО chat_input, чтобы он остался сверху,
        # а наполняем после — тогда ответ появляется в том же прогоне скрипта
        box = st.container(height=700, key="chat_box")
        user_input = st.chat_input("Напишите вопрос…")

        if user_input and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input.strip()})
            st.session_state.last_error = None
            st.session_state.pending = True

        with box:
            if not st.session_state.messages:
                with st.chat_message("assistant"):
                    st.markdown(GREETING)
            for idx, msg in enumerate(st.session_state.messages):
                _render_message(idx, msg)

            if st.session_state.pending:
                _answer_pending()
            if st.session_state.last_error:
                _render_error()
