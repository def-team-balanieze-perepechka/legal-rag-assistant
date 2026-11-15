"""
Мост между Streamlit и api/client.py: клиент, ожидание готовности,
подготовка истории, перевод ошибок API в понятный пользователю текст.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor

import streamlit as st

from api.client import LegalAssistantClient, LegalAssistantError

API_URL = os.getenv("API_URL", "http://localhost:8000")   # в docker compose задаётся в environment
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
    st.caption("Обычно это занимает до минуты после запуска.")
    if DEBUG:
        st.caption(f"API: {API_URL}")
    time.sleep(5)
    st.rerun()


def log_error(context: str) -> None:
    """Полный traceback — в консоль/логи контейнера, пользователю он не показывается."""
    print(f"[ui] {context}\n{traceback.format_exc()}", file=sys.stderr, flush=True)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_documents() -> list[dict]:
    return get_client().documents()          # при ошибке исключение -> результат не кэшируется


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_document(doc_id: str) -> dict:
    return get_client().document(doc_id)


def load_documents() -> list[dict]:
    try:
        return _fetch_documents()
    except Exception:                         # сайдбар не должен ронять приложение
        log_error("не удалось загрузить список документов")
        return []


def load_document(doc_id: str) -> dict | None:
    try:
        return _fetch_document(doc_id)
    except Exception:                         # в т.ч. AttributeError, если в образе старый api/client.py
        log_error(f"не удалось загрузить документ {doc_id}")
        return None


def build_payload(messages: list[dict]) -> list[dict]:
    """Только role+content (без status/citations/debug), последние HISTORY_LIMIT реплик."""
    clean = [{"role": m["role"], "content": m["content"]} for m in messages]
    return clean[-HISTORY_LIMIT:]


@st.cache_resource
def get_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="ui-chat")


def _chat_call(client: LegalAssistantClient, payload: list[dict], allow_clarify: bool) -> dict:
    """Выполняется в ФОНОВОМ потоке: никаких st.* и session_state здесь."""
    data = client.chat(messages=payload, allow_clarify=allow_clarify)
    if data.get("status") == "error":
        # ядро вернуло заглушку FAIL_ANSWER с HTTP 200 — для UI это ошибка, не ответ
        raise LegalAssistantError("ядро не смогло сформировать ответ", status_code=502)
    return data


def submit_chat(messages: list[dict]) -> Future:
    """
    Отправляет запрос в фон и сразу возвращает Future. Вызывать из потока скрипта/колбэка:
    session_state и клиент читаются здесь, а в поток уходят только готовые данные.
    Благодаря этому скрипт не стоит в ожидании ответа, и остальные клики (документы, источники) обрабатываются сразу.
    """
    return get_client().chat_async(
        messages=build_payload(messages),
        allow_clarify=bool(st.session_state.get("allow_clarify", False)))


def describe_error(exc: LegalAssistantError) -> tuple[str, bool]:
    """-> (текст для пользователя, можно ли повторить). Технические детали — только в лог (и в UI при UI_DEBUG=1)."""
    code = exc.status_code
    # ожидаемая ошибка API — одна строка в лог, без traceback (traceback пишем только для неожиданных исключений)
    print(f"[ui] ошибка API: HTTP {code}: {exc.detail} (API_URL={API_URL})", file=sys.stderr, flush=True)
    if code == 400:
        text, retry = f"Некорректный запрос: {exc.detail}", False
    elif code == 422:
        text, retry = "Ошибка интерфейса: неверный формат запроса. Подробности в логах.", False
    elif code == 502:
        text, retry = "Ассистент не смог сформировать ответ: модель недоступна или вернула некорректный ответ. Повторите позже.", True
    elif code == 503:
        text, retry = "Сервис ещё загружает базу знаний. Повторите через 10 секунд.", True
    elif code == 504:
        text, retry = "Ассистент не успел ответить (таймаут). Повторите или упростите вопрос.", True
    elif code is None:
        text, retry = "Нет связи с сервером ассистента. Проверьте, что бэкенд запущен, и повторите.", True
    else:
        text, retry = f"Сервис временно недоступен (код {code}). Повторите позже.", True
    if DEBUG:
        text += f"\n\n`{code}: {exc.detail}`"
    return text, retry


GENERIC_ERROR = "Ошибка обработки запроса. Попробуйте ещё раз; если не помогает — нажмите «Новый диалог»."
