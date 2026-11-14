"""
Мост между Streamlit и api/client.py: клиент, ожидание готовности,
подготовка истории, перевод ошибок API в понятный пользователю текст.
"""
from __future__ import annotations

import os
import sys
import time

import streamlit as st

from api.client import LegalAssistantClient, LegalAssistantError

API_URL = os.getenv("API_URL", "http://localhost:8000")
# Сколько последних реплик отправлять в API. Ядро строит поисковый контекст из ВСЕХ
# реплик пользователя в истории, поэтому длинный диалог «размывает» поиск и раздувает промпт.
HISTORY_LIMIT = int(os.getenv("UI_HISTORY_LIMIT", "10"))
DEBUG = os.getenv("UI_DEBUG", "").strip().lower() in ("1", "true", "yes")


@st.cache_resource
def get_client() -> LegalAssistantClient:
    return LegalAssistantClient(API_URL)


def wait_until_ready() -> None:
    """Блокирует отрисовку, пока API не ответит ok. После первого успеха больше не опрашивает."""
    if st.session_state.get("api_ready"):
        return
    if get_client().is_ready():
        st.session_state.api_ready = True
        return
    st.warning("⏳ Сервис загружает базу знаний, подождите…")
    st.caption(f"API: {API_URL}")
    time.sleep(5)
    st.rerun()


@st.cache_data(ttl=300, show_spinner=False)
def load_documents() -> list[dict]:
    try:
        return get_client().documents()
    except LegalAssistantError:
        return []   # сайдбар не должен ронять приложение


@st.cache_data(ttl=600, show_spinner=False)
def load_document(doc_id: str) -> dict | None:
    try:
        return get_client().document(doc_id)
    except LegalAssistantError:
        return None


def build_payload(messages: list[dict]) -> list[dict]:
    """Только role+content (без status/citations/debug), последние HISTORY_LIMIT реплик."""
    clean = [{"role": m["role"], "content": m["content"]} for m in messages]
    return clean[-HISTORY_LIMIT:]


def ask(messages: list[dict]) -> dict:
    """Режим B (messages). Бросает LegalAssistantError."""
    data = get_client().chat(
        messages=build_payload(messages),
        allow_clarify=bool(st.session_state.get("allow_clarify", False)),
    )
    if data.get("status") == "error":
        # ядро вернуло заглушку FAIL_ANSWER с HTTP 200 — для UI это ошибка, не ответ
        raise LegalAssistantError("ядро не смогло сформировать ответ", status_code=502)
    return data


def describe_error(exc: LegalAssistantError) -> tuple[str, bool]:
    """-> (текст для пользователя, можно ли повторить)."""
    code = exc.status_code
    if code == 400:
        return f"Некорректный запрос: {exc.detail}", False
    if code == 422:
        print(f"[ui] 422 от API (баг формата запроса): {exc.detail}", file=sys.stderr)
        return "Ошибка интерфейса: неверный формат запроса. Подробности в консоли.", False
    if code == 503:
        return "Сервис ещё загружает базу знаний. Повторите через 10 секунд.", True
    if code == 504:
        return "Ассистент не успел ответить (таймаут). Повторите или упростите вопрос.", True
    if code is None:
        return f"Нет связи с API ({API_URL}). Проверьте, что бэкенд запущен.", True
    return "Сервис временно недоступен (ошибка модели или сети). Повторите позже.", True
