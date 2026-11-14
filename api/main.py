"""
FastAPI-бекенд юридического ассистента.

Запуск из корня проекта:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Контракт (стабильный):
    GET  /health    — статус сервиса и БЗ (без LLM; 200=ok, 503=starting)
    GET  /documents — каталог документов для UI
    POST /chat      — основной вопрос/диалог

Режимы POST /chat (по приоритету):
    messages > question + history > question
"""
from __future__ import annotations

#
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_CORE = _ROOT / "core"
if _CORE.is_dir() and str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
# --- /bootstrap ---

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Literal

from dotenv import load_dotenv
load_dotenv()  # .env из корня; в Docker переменные приходят через env_file

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from assistant import Assistant
from llm import LLMClient


# ---------- схемы ----------

from pydantic import field_validator

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

    @field_validator("content")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content не может быть пустым или из пробелов")
        return v


class ChatRequest(BaseModel):
    question: str | None = Field(None, description="Одиночный вопрос или диалог одной строкой")
    messages: list[Message] | None = Field(None, description="Нативный чат; последнее — от user")
    history: list[Message] | None = Field(None, description="Отдельная история при явном question")
    allow_clarify: bool = False


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    short_ref: str
    ref: str
    quote: str
    edition: str | None = None
    status: str | None = None


class ChatResponse(BaseModel):
    status: str
    answer: str
    citations: list[Citation] = []
    missing_info: str | None = None
    debug: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "starting"]
    documents: int = 0
    chunks: int = 0
    dense: bool = False


# ---------- конфиг ----------

def resolve_kb_path() -> str:
    """KB_PATH → иначе DATA_DIR/knowledge-base.zip → иначе DATA_DIR."""
    explicit = os.getenv("KB_PATH", "").strip()
    if explicit:
        return explicit
    p = Path(os.getenv("DATA_DIR", "data"))
    if p.is_dir():
        zp = p / "knowledge-base.zip"
        return str(zp if zp.exists() else p)
    return str(p)


CHAT_TIMEOUT = float(os.getenv("CHAT_TIMEOUT", "180"))

_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()] or ["*"]


# ---------- lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    kb_path = resolve_kb_path()
    print(f"[api] загрузка БЗ: {kb_path}")
    client = LLMClient()
    assistant = Assistant.from_path(kb_path, client)
    # build_index блокирующий → в threadpool, чтобы не вешать startup
    await run_in_threadpool(assistant.build_index)
    print(f"[api] БЗ загружена: документов={len(assistant.kb.documents)}, "
          f"чанков={len(assistant.kb.chunks)}, dense={assistant.dense_ready}")
    app.state.assistant = assistant
    app.state.client = client
    app.state.ready = True
    try:
        yield
    finally:
        client.close()
        print("[api] остановлен")


app = FastAPI(title="Legal Assistant API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- dependencies ----------

def require_ready(request: Request) -> None:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(503, "Сервис инициализируется, повторите через 10 секунд")


# ---------- ручки ----------

@app.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    """503 пока БЗ грузится — Docker healthcheck будет ждать, пока сервис не готов."""
    if not getattr(request.app.state, "ready", False):
        response.status_code = 503
        return HealthResponse(status="starting")
    a: Assistant = request.app.state.assistant
    return HealthResponse(
        status="ok",
        documents=len(a.kb.documents),
        chunks=len(a.kb.chunks),
        dense=a.dense_ready,
    )


@app.get("/documents", dependencies=[Depends(require_ready)])
def list_documents(request: Request) -> list[dict]:
    a: Assistant = request.app.state.assistant
    return [
        {
            "doc_id": d.doc_id,
            "doc_type": d.doc_type,
            "title": d.title,
            "short_ref": d.short_ref,
            "number": d.number,
            "date": d.date.isoformat() if d.date else None,
            "status": d.status,
            "valid_from": d.valid_from.isoformat() if d.valid_from else None,
            "valid_to": d.valid_to.isoformat() if d.valid_to else None,
        }
        for d in a.kb.documents.values()
    ]


def _extract_question(req: ChatRequest) -> tuple[str, list[dict] | None]:
    """Приоритет: messages > question+history > question. Пустые строки — 400."""
    def _check(q: str) -> str:
        q = (q or "").strip()
        if not q:
            raise HTTPException(400, "пустой вопрос: question/content не должен быть пустым или из пробелов")
        return q

    if req.messages:
        msgs = [m.model_dump() for m in req.messages]
        if msgs[-1]["role"] != "user":
            raise HTTPException(400, "последнее сообщение в messages должно быть от user")
        return _check(msgs[-1]["content"]), (msgs[:-1] or None)
    if req.question and req.history:
        return _check(req.question), [m.model_dump() for m in req.history]
    if req.question:
        return _check(req.question), None
    raise HTTPException(400, "нужен либо messages, либо question")


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_ready)])
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    assistant: Assistant = request.app.state.assistant
    question, history = _extract_question(req)

    try:
        result = await asyncio.wait_for(
            run_in_threadpool(assistant.answer, question, history, req.allow_clarify),
            timeout=CHAT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, f"превышен таймаут {CHAT_TIMEOUT:.0f} с")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"ошибка ассистента: {type(exc).__name__}: {exc}") from exc

    citations = [
        Citation(
            chunk_id=c.get("chunk_id", ""),
            doc_id=c.get("doc_id", ""),
            title=c.get("title", ""),
            short_ref=c.get("short_ref", ""),
            ref=c.get("ref", ""),
            quote=c.get("quote", ""),
            edition=c.get("edition"),
            status=c.get("status"),
        )
        for c in (result.get("citations") or [])
    ]
    return ChatResponse(
        status=result.get("status", "answer"),
        answer=result.get("answer", ""),
        citations=citations,
        missing_info=result.get("missing_info"),
        debug=result.get("debug"),
    )