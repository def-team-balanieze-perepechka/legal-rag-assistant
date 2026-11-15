"""
Диалог с ассистентом: история из session_state.messages, фоновый запрос к API, рендер ответов.

Формат сообщения в session_state.messages:
  user:      {"role": "user", "content": str}
  assistant: {"role": "assistant", "content": str, "status": str,
              "citations": list[dict], "missing_info": str | None, "debug": dict | None}

Как устроено ожидание ответа
----------------------------
Запрос к API уходит в фоновый поток (ui_api.submit_chat), а скрипт сразу завершается. Пока ответа нет,
`_job_view` (st.fragment с run_every=1) раз в секунду проверяет Future и рисует «Ищу в базе знаний… N с».
Поэтому остальные действия — открыть документ, «Источники», «Новый диалог» — работают во время ожидания.
Раньше запрос шёл прямо в потоке скрипта: Streamlit не может обработать новый клик, пока скрипт стоит
внутри блокирующего вызова, и все клики ждали ответа.

session_state.job = {"future": Future, "t0": float, "result": None | {...}}
  result заполняется, когда ответ забран (фрагментом или _settle_job); следующий полный прогон скрипта обнуляет job
  (ответ к этому моменту уже в истории и рисуется обычным циклом).

Поле ввода живёт в собственном фрагменте (`_input_area`) и заблокировано, пока запрос не завершился
(ответ, ошибка — любой исход). Фрагмент опрашивает состояние раз в секунду, поэтому блокировка снимается
без полного перерисовывания страницы (открытый диалог документа при этом не закрывается).

Если запрос завершился ошибкой, неудавшийся вопрос остаётся в чате только до следующего вопроса:
он нужен для «Повторить». Новый вопрос убирает его из чата и из истории, уходящей в API.
"""
import re
import time

import streamlit as st

from api.client import LegalAssistantError
from ui_api import DEBUG, GENERIC_ERROR, describe_error, log_error, submit_chat

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


# ---------- колбэки и запуск запроса ----------

def _open_sources(idx: int) -> None:
    st.session_state.selected_msg = idx


def _start_job() -> None:
    """Отправляет текущую историю в фон. Старый незавершённый запрос (если был) просто теряет значение."""
    st.session_state.pending = False
    st.session_state.last_error = None
    st.session_state.job = {"future": submit_chat(st.session_state.messages), "t0": time.time(), "result": None}


def _settle_job() -> None:
    """
    Запрос уже завершён, а фрагмент ещё не успел забрать результат (опрос раз в секунду) — забираем здесь.
    Нужно, чтобы ответ попал в историю ДО нового вопроса и порядок сообщений не нарушился.
    """
    job = st.session_state.get("job")
    if job is not None and job["result"] is None and job["future"].done():
        job["result"] = _consume(job["future"])


def _submit_question(text: str) -> bool:
    """Принимает новый вопрос. False — если предыдущий запрос ещё идёт (поле ввода в этот момент заблокировано)."""
    _settle_job()
    job = st.session_state.get("job")
    if job is not None:
        if job["result"] is None:
            return False
        st.session_state.job = None                   # завершённый запрос: ответ уже в истории либо ошибка

    messages = st.session_state.messages
    if st.session_state.get("last_error") and messages and messages[-1]["role"] == "user":
        messages.pop()                                # неудавшийся вопрос не остаётся ни в чате, ни в истории для API
    st.session_state.last_error = None

    messages.append({"role": "user", "content": text})
    st.session_state.pending = True
    return True


# ---------- рендер сообщений ----------

def _render_extras(idx: int, msg: dict, in_fragment: bool = False) -> None:
    """
    Плашки по статусу, кнопка источников, debug — всё, что идёт после текста ответа.

    in_fragment=True: клик по кнопке внутри st.fragment перезапускает только фрагмент, а панель источников
    стоит вне его — поэтому такая кнопка сама вызывает полный st.rerun().
    """
    status = msg.get("status", "answer")
    if status == "insufficient":
        st.info(f"**Чего не хватает в базе:** {msg.get('missing_info') or '—'}")
    elif status in STATUS_NOTES:
        kind, text = STATUS_NOTES[status]
        getattr(st, kind)(text)

    citations = msg.get("citations") or []
    if citations:
        label = f"📎 Источники ({len(citations)})"
        if in_fragment:
            if st.button(label, key=f"src_{idx}"):
                _open_sources(idx)
                st.rerun()
        else:
            st.button(label, key=f"src_{idx}", on_click=_open_sources, args=(idx,))
    if DEBUG and msg.get("debug"):
        with st.expander("Debug"):
            st.json(msg["debug"])


def _render_message(idx: int, msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("\n", "  \n"))
        if msg["role"] == "assistant":
            _render_extras(idx, msg)


def _typewriter(text: str):
    """Посимвольный «набор» уже полученного ответа. Это визуальный эффект, а не стриминг модели."""
    parts = re.findall(r"\s+|\S+", text.replace("\n", "  \n"))
    delay = min(0.03, 3.0 / max(len(parts), 1))      # весь эффект — не дольше ~3 секунд
    for part in parts:
        yield part
        time.sleep(delay)


def _render_error(slot) -> None:
    err = st.session_state.last_error
    with slot.container():
        with st.chat_message("assistant"):
            st.error(err["text"])
            if err["retryable"] and st.button("🔄 Повторить", key="retry_btn"):
                # полный rerun (не только фрагмента): фрагмент должен пересоздаться с опросом run_every=1
                _start_job()
                st.rerun()


# ---------- ожидание ответа (фрагмент) ----------

def _consume(fut) -> dict:
    """Забирает результат завершённого Future. Любая ошибка -> сообщение пользователю, не traceback."""
    try:
        data = fut.result()
        msg = {
            "role": "assistant",
            "content": data["answer"],
            "status": data.get("status", "answer"),
            "citations": data.get("citations") or [],
            "missing_info": data.get("missing_info"),
            "debug": data.get("debug"),
        }
    except LegalAssistantError as exc:
        text, retryable = describe_error(exc)
        st.session_state.last_error = {"text": text, "retryable": retryable}
        return {"kind": "error"}
    except Exception:                                # неожиданный формат ответа, старый client.py и т.п.
        log_error("неожиданная ошибка при обработке ответа API")
        st.session_state.last_error = {"text": GENERIC_ERROR, "retryable": True}
        return {"kind": "error"}

    st.session_state.messages.append(msg)
    return {"kind": "msg", "idx": len(st.session_state.messages) - 1, "fresh": True}


def _job_view() -> None:
    """
    Рисует «Ищу…» / только что пришедший ответ / ошибку. Слоты error_slot и work_slot стоят на одних и тех же путях
    в каждом прогоне: так прошлая ошибка гарантированно исчезает, когда на её месте появляется спиннер
    (Streamlit иначе держит старые дочерние элементы переиспользованного блока как «устаревшие»).
    """
    error_slot = st.empty()
    work_slot = st.empty()
    job = st.session_state.get("job")

    if job is None:                                   # запроса нет: показать ошибку прошлого, если есть
        if st.session_state.get("last_error"):
            _render_error(error_slot)
        return

    fut = job["future"]
    if not fut.done():
        with work_slot.container():
            with st.chat_message("assistant"):
                st.markdown(f"⏳ Ищу в базе знаний… {int(time.time() - job['t0'])} с")
        return

    if job["result"] is None:                         # первый тик после завершения: забрать ответ
        job["result"] = _consume(fut)
    res = job["result"]

    if res["kind"] == "error":
        _render_error(error_slot)
        return

    msg = st.session_state.messages[res["idx"]]
    with work_slot.container():
        with st.chat_message("assistant"):
            if res.pop("fresh", False):               # эффект набора — только при первом показе
                st.write_stream(_typewriter(msg["content"]))
            else:
                st.markdown(msg["content"].replace("\n", "  \n"))
            _render_extras(res["idx"], msg, in_fragment=True)


def _input_area() -> None:
    """Поле ввода + переключатель. Заблокировано, пока запрос не завершится любым исходом."""
    job = st.session_state.get("job")
    busy = job is not None and not job["future"].done()

    user_input = st.chat_input("Напишите вопрос…", disabled=busy)
    st.toggle(
        "Уточняющие вопросы",
        key="allow_clarify",
        help="Если из вопроса неясно, о каком договоре или периоде речь, "
             "ассистент сначала уточнит, а потом ответит.",
    )
    if user_input and user_input.strip():
        if _submit_question(user_input.strip()):
            st.rerun()                                # полный rerun: запрос запускается и рисуется в основном скрипте


def render_chat_content(chat_col) -> None:
    with chat_col:
        st.subheader("💬 Вопрос по базе знаний")

        _settle_job()                                 # завершившийся, но не забранный ответ — в историю до отрисовки
        if st.session_state.pending:                  # вопрос принят в _input_area
            _start_job()

        job = st.session_state.get("job")
        running = job is not None and job["result"] is None   # опрос дважды в секунду — только пока запрос в работе
        poll = 0.5 if running else None

        # контейнер создаём ДО поля ввода, чтобы он остался сверху, а наполняем после
        box = st.container(height=700, key="chat_box")
        st.fragment(_input_area, run_every=poll)()

        with box:
            if job is not None and job["result"] is not None:
                st.session_state.job = None           # ответ уже в истории — рисуется циклом ниже

            if not st.session_state.messages:
                with st.chat_message("assistant"):
                    st.markdown(GREETING)
            for idx, msg in enumerate(st.session_state.messages):
                _render_message(idx, msg)

            st.fragment(_job_view, run_every=poll)()
