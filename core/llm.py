from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass

import httpx

RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(env(name))))
    except ValueError:
        return default


def env_float(name: str, default: float, low: float = 0.0, high: float = 2.0) -> float:
    try:
        return max(low, min(high, float(env(name))))
    except ValueError:
        return default


@dataclass
class Settings:
    base_url: str
    api_key: str
    model: str
    emb_model: str
    emb_base_url: str
    emb_key: str
    rerank_model: str
    rerank_base_url: str
    rerank_key: str
    temperature: float
    max_tokens: int
    extra_body: dict
    timeout: float

    @classmethod
    def from_env(cls) -> "Settings":
        base = (env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")).rstrip("/")
        key = env("LLM_API_KEY") or env("OPENAI_API_KEY")
        try:
            extra = json.loads(env("LLM_EXTRA_BODY", "{}"))
        except json.JSONDecodeError:
            print("[llm] LLM_EXTRA_BODY не JSON — игнорируем", file=sys.stderr)
            extra = {}
        return cls(
            base_url=base,
            api_key=key,
            model=env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL", "deepseek/deepseek-v4.1-flash"),
            emb_model=env("EMB_MODEL", "baai/bge-m3"),
            emb_base_url=(env("DEV_EMBED_BASE_URL") or base).rstrip("/"),
            emb_key=env("EMBEDDER_API_KEY") or key,
            rerank_model=env("RERANKER_MODEL", "cohere/rerank-5-fast"),
            rerank_base_url=(env("DEV_RERANK_BASE_URL") or base).rstrip("/"),
            rerank_key=env("RERANKER_API_KEY") or key,
            temperature=env_float("TEMPERATURE", 0.0),
            max_tokens=env_int("MAX_TOKENS", 1024, 128, 8192),
            extra_body=extra if isinstance(extra, dict) else {},
            timeout=env_float("LLM_TIMEOUT", 90.0, 5.0, 600.0),
        )

    def describe(self) -> str:
        """Для логов: без ключей."""
        return (f"model={self.model} emb={self.emb_model} rerank={self.rerank_model} "
                f"temperature={self.temperature} max_tokens={self.max_tokens} "
                f"extra_body={list(self.extra_body)} dev_embed={self.emb_base_url != self.base_url} "
                f"dev_rerank={self.rerank_base_url != self.base_url}")


class ServiceDisabled(RuntimeError):
    """
    Сервис (эмбеддер / реранкер) отключен после неустранимой ошибки
    не дергаем его снова.
    """


class LLMClient:
    MAX_FAILURES = 3

    def __init__(self, settings: Settings | None = None, attempts: int = 3) -> None:
        self.s = settings or Settings.from_env()
        self.attempts = attempts
        self.http = httpx.Client(timeout=self.s.timeout)
        self._disabled: dict[str, str] = {}
        self._failures: dict[str, int] = {}
        self._lock = threading.Lock()
        self.calls = {"chat": 0, "embed": 0, "rerank": 0}

    def close(self) -> None:
        self.http.close()

    # Транспорт

    def _post(self, base: str, path: str, key: str, payload: dict, retry_timeouts: bool = True) -> dict:
        if not base:
            raise ValueError("не задан LLM_BASE_URL")
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                r = self.http.post(base + path, headers={"Authorization": f"Bearer {key}"}, json=payload)
                if r.status_code in RETRY_STATUSES:
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as exc:
                last = exc
                if exc.response.status_code not in RETRY_STATUSES:
                    raise
            except httpx.ReadTimeout:
                # Сервер принял запрос, но не успел ответить
                # Повтор заново посчитает тот же промпт
                if not retry_timeouts:
                    raise
                last = sys.exc_info()[1]
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
            if attempt + 1 < self.attempts:
                time.sleep(min(8.0, 1.0 * 2 ** attempt))
        assert last is not None
        raise last

    def _guard(self, service: str) -> None:
        if service in self._disabled:
            raise ServiceDisabled(self._disabled[service])

    def _ok(self, service: str) -> None:
        self._failures[service] = 0

    def _disable(self, service: str, exc: Exception) -> None:
        """
        Необязательные сервисы (эмбеддер, реранкер) отключаем, чтобы не тратить время
        на каждом вопросе: сразу — при 4xx (эндпоинта нет или модель не та),
        иначе — после MAX_FAILURES неудач подряд.
        """
        with self._lock:
            self._failures[service] = self._failures.get(service, 0) + 1
            hard = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in RETRY_STATUSES
            if hard or self._failures[service] >= self.MAX_FAILURES:
                reason = f"HTTP {exc.response.status_code}" if hard else f"{self._failures[service]} ошибки подряд"
                if service not in self._disabled:
                    print(f"[llm] {service} отключён до конца прогона: {reason}", file=sys.stderr)
                self._disabled[service] = f"{service}: {reason}"

    # API

    def chat(self, messages: list[dict], max_tokens: int | None = None) -> str:
        payload = {"model": self.s.model, "messages": messages, "temperature": self.s.temperature,
                   "max_tokens": max_tokens or self.s.max_tokens, **self.s.extra_body}
        self.calls["chat"] += 1
        data = self._post(self.s.base_url, "/chat/completions", self.s.api_key, payload, retry_timeouts=False)
        content = data["choices"][0]["message"].get("content") or ""
        return strip_thinking(content)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._guard("embed")
        self.calls["embed"] += 1
        try:
            data = self._post(self.s.emb_base_url, "/embeddings", self.s.emb_key,
                              {"model": self.s.emb_model, "input": texts, "encoding_format": "float"})
        except Exception as exc:
            self._disable("embed", exc)
            raise
        self._ok("embed")
        out: list[list[float] | None] = [None] * len(texts)
        for pos, item in enumerate(data["data"]):
            out[int(item.get("index", pos))] = [float(x) for x in item["embedding"]]
        if any(v is None for v in out):
            raise ValueError("получены не все эмбеддинги")
        return out  # type: ignore[return-value]

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """Список (индекс документа, score) по убыванию релевантности."""
        self._guard("rerank")
        self.calls["rerank"] += 1
        try:
            data = self._post(self.s.rerank_base_url, "/rerank", self.s.rerank_key,
                              {"model": self.s.rerank_model, "query": query, "documents": documents,
                               "top_n": min(top_n, len(documents))})
        except Exception as exc:
            self._disable("rerank", exc)
            raise
        self._ok("rerank")
        items = data.get("results", data.get("data", []))
        out = []
        for rank, it in enumerate(items):
            score = it.get("relevance_score", it.get("score"))
            out.append((int(it["index"]), float(score) if score is not None else -float(rank)))
        return out


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()
