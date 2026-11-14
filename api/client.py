"""
HTTP-клиент юридического ассистента. Используется UI (Streamlit и др.).

    from api.client import LegalAssistantClient, LegalAssistantError

    client = LegalAssistantClient()            # API_URL или http://localhost:8000
    if not client.is_ready():
        raise RuntimeError("сервис ещё грузится")

    data = client.chat(messages=[
        {"role": "user", "content": "Что за договор П-17/2024?"},
    ])
    print(data["answer"], data["citations"])
"""
from __future__ import annotations

import os
from typing import Any

import httpx


class LegalAssistantError(RuntimeError):
    """Единый тип ошибки — UI ловит его, а не кишки httpx."""


class LegalAssistantClient:
    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        self.base = (base_url or os.getenv("API_URL", "http://localhost:8000")).rstrip("/")
        self.timeout = timeout

    def __enter__(self) -> "LegalAssistantClient":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    # ---------- транспорт ----------

    def _request(self, method: str, path: str, *, expect: tuple[int, ...] = (200,), **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", self.timeout)
        try:
            r = httpx.request(method, f"{self.base}{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise LegalAssistantError(f"сеть: {type(exc).__name__}: {exc}") from exc

        if r.status_code in expect:
            try:
                return r.json()
            except ValueError:
                raise LegalAssistantError(f"не JSON в ответе {path}: {r.text[:200]}")
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise LegalAssistantError(f"HTTP {r.status_code}: {detail}")

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