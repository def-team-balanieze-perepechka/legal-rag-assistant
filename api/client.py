"""
HTTP-клиент юридического ассистента. Используется UI (Streamlit и др.).

    from api.client import LegalAssistantClient, LegalAssistantError

    client = LegalAssistantClient()            # API_URL или http://localhost:8000
    if not client.is_ready():
        raise RuntimeError("сервис ещё грузится")

    data = client.chat(messages=[{"role": "user", "content": "Что за договор П-17/2024?"}])
    print(data["answer"], data["citations"])
"""
from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import Future
from typing import Any
from urllib.parse import quote

import httpx

def _env_float(name: str, default: float) -> float:
    """float из окружения; пустое или битое значение (например `CHAT_TIMEOUT=` в .env) -> default, а не падение при импорте."""
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


# Должен быть БОЛЬШЕ серверного CHAT_TIMEOUT (по умолчанию 180 с): сервер сам обрывает запрос на CHAT_TIMEOUT
# и отвечает 504; клиент, ждущий меньше, увидел бы вместо него «нет связи».
DEFAULT_TIMEOUT = _env_float("CHAT_TIMEOUT", 180.0) + 30.0


class LegalAssistantError(RuntimeError):
    """
    Единый тип ошибки — UI ловит его, а не кишки httpx.

    status_code — HTTP-код ответа API (400/422/503/504/5xx);
                  None, если до API достучаться не удалось (сеть, таймаут клиента).
    detail      — текст из поля "detail" ответа API.
    """

    def __init__(self, message: str, status_code: int | None = None, detail: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail or message


class LegalAssistantClient:
    def __init__(self, base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT):
        self.base = (base_url or os.getenv("API_URL", "http://localhost:8000")).rstrip("/")
        self.timeout = timeout
        # один Client = переиспользование соединений; потокобезопасен
        # connect=5: если хост недоступен, «нет связи» появляется за секунды, а не через весь read-таймаут
        self._http = httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))
        # фоновый event loop для отменяемых запросов (chat_async); создаётся лениво
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ahttp: httpx.AsyncClient | None = None
        self._loop_lock = threading.Lock()

    def close(self) -> None:
        self._http.close()
        if self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._ahttp.aclose(), self._loop).result(timeout=5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None

    def __enter__(self) -> "LegalAssistantClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------- транспорт ----------

    def _handle(self, r: httpx.Response, path: str, expect: tuple[int, ...] = (200,)) -> Any:
        """Разбор ответа — общий для sync- и async-запросов."""
        if r.status_code in expect:
            try:
                return r.json()
            except ValueError as exc:
                raise LegalAssistantError(
                    f"не JSON в ответе {path}: {r.text[:200]}", status_code=r.status_code
                ) from exc
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        if not isinstance(detail, str):          # 422: detail — список ошибок валидации
            detail = str(detail)
        raise LegalAssistantError(f"HTTP {r.status_code}: {detail}", status_code=r.status_code, detail=detail)

    def _request(self, method: str, path: str, *, expect: tuple[int, ...] = (200,), **kwargs: Any) -> Any:
        try:
            r = self._http.request(method, f"{self.base}{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise LegalAssistantError(f"сеть: {type(exc).__name__}: {exc}") from exc
        return self._handle(r, path, expect)

    # ---------- отменяемый /chat (для UI) ----------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._loop_lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, daemon=True, name="client-http-loop").start()

                async def _make() -> httpx.AsyncClient:
                    return httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0))

                self._ahttp = asyncio.run_coroutine_threadsafe(_make(), loop).result()
                self._loop = loop
            return self._loop

    async def _achat(self, payload: dict) -> dict:
        try:
            r = await self._ahttp.post(f"{self.base}/chat", json=payload)
        except httpx.HTTPError as exc:
            raise LegalAssistantError(f"сеть: {type(exc).__name__}: {exc}") from exc
        return self._handle(r, "/chat")

    def chat_async(self, *, messages: list[dict], allow_clarify: bool = False) -> Future:
        """
        Запускает /chat в фоне и возвращает concurrent.futures.Future: .done(), .result(), .cancel().
        cancel() обрывает HTTP-запрос (сервер видит разрыв соединения).
        """
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(
            self._achat({"messages": messages, "allow_clarify": allow_clarify}), loop
        )

    # ---------- API ----------

    def health(self) -> dict:
        """200 (ok) или 503 (starting) — оба считаются валидным ответом."""
        return self._request("GET", "/health", expect=(200, 503), timeout=10)

    def is_ready(self) -> bool:
        try:
            return self.health().get("status") == "ok"
        except LegalAssistantError:
            return False

    def documents(self) -> list[dict]:
        return self._request("GET", "/documents", timeout=30)

    def document(self, doc_id: str) -> dict:
        """Полный текст и метаданные документа (GET /documents/{doc_id})."""
        return self._request("GET", f"/documents/{quote(doc_id, safe='')}", timeout=30)

    def chat(
        self,
        *,
        question: str | None = None,
        messages: list[dict] | None = None,
        history: list[dict] | None = None,
        allow_clarify: bool = False,
    ) -> dict:
        """
        Ровно один из режимов:
          chat(messages=[...])                  — нативный чат (рекомендуется для UI)
          chat(question="...", history=[...])   — если UI ведёт историю сам
          chat(question="...")                  — одиночный вопрос / диалог одной строкой
        """
        payload: dict[str, Any] = {"allow_clarify": allow_clarify}
        if messages is not None:
            payload["messages"] = messages
        elif question is not None:
            payload["question"] = question
            if history is not None:
                payload["history"] = history
        else:
            raise ValueError("нужен либо messages, либо question")
        return self._request("POST", "/chat", json=payload)
