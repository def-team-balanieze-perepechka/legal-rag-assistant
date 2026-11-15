"""Запуск (из корня проекта):  streamlit run ui/app.py"""
import sys
from pathlib import Path

# Streamlit добавляет в sys.path папку скрипта (ui/), а не корень проекта —
# без этого `from api.client import ...` не найдётся.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# set_page_config обязан быть ПЕРВОЙ командой Streamlit (раньше inject_css -> st.html)
st.set_page_config(
    page_title="Юридический ассистент",
    page_icon=str(Path(__file__).parent / "icons" / "favicon.png"),
    layout="wide",
)

from styles.loader import inject_css      # noqa: E402
from ui_api import wait_until_ready       # noqa: E402
from ui_chat import render_chat_content   # noqa: E402
from ui_docview import show_pending_document  # noqa: E402
from ui_refpanel import render_reference_panel  # noqa: E402
from ui_sidebar import render_sidebar     # noqa: E402

inject_css()

# --- состояние (один диалог; бэкенд stateless) ---
st.session_state.setdefault("messages", [])       # история диалога
st.session_state.setdefault("pending", False)     # ждём ответ API на последнее user-сообщение
st.session_state.setdefault("last_error", None)   # {"text": str, "retryable": bool}
st.session_state.setdefault("selected_msg", None) # индекс ответа, чьи источники открыты справа

wait_until_ready()
render_sidebar()

if st.session_state.selected_msg is not None:
    _, chat_col, panel_col = st.columns([0.3, 2, 1.5], gap="large")
else:
    _, chat_col, _ = st.columns([1, 2, 1], gap="large")
    panel_col = None

render_chat_content(chat_col)
render_reference_panel(panel_col)
show_pending_document()   # диалог просмотра документа (если запрошен кнопкой)
