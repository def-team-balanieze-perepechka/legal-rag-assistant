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

import os
from typing import Any
from urllib.parse import quote

import httpx

# Должен быть БОЛЬШЕ серверного CHAT_TIMEOUT (180 с), иначе клиент оборвёт
# соединение раньше, чем сервер успеет вернуть осмысленный 504.
DEFAULT_TIMEOUT = API_URL = float(os.getenv("LLM_TIMEOUT", 300.0)) + 10.0


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
        self._http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "LegalAssistantClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------- транспорт ----------

    def _request(self, method: str, path: str, *, expect: tuple[int, ...] = (200,), **kwargs: Any) -> Any:
        try:
            r = self._http.request(method, f"{self.base}{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise LegalAssistantError(f"сеть: {type(exc).__name__}: {exc}") from exc

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
