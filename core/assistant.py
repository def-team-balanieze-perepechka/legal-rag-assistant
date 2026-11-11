from __future__ import annotations

import re
import time

import generate as gen
from kb import load_kb
from llm import LLMClient, env_int
from search import Retriever

_TURN_RE = re.compile(r"^\s*(Пользователь|Ассистент|User|Assistant)\s*:\s*", re.I | re.M)
_ROLES = {"пользователь": "user", "user": "user", "ассистент": "assistant", "assistant": "assistant"}


def parse_dialog(text: str) -> tuple[str, list[dict]]:
    """
    «Пользователь: … / Ассистент: … / Пользователь: …» ->
    (последний вопрос, история до него).
    """
    marks = list(_TURN_RE.finditer(text))
    if not marks:
        return text.strip(), []
    turns = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        content = text[m.end():end].strip()
        if content:
            turns.append({"role": _ROLES[m.group(1).lower()], "content": content})
    prefix = text[:marks[0].start()].strip()
    if prefix:
        turns.insert(0, {"role": "user", "content": prefix})
    last_user = max((i for i, t in enumerate(turns) if t["role"] == "user"), default=None)
    if last_user is None:
        return text.strip(), []
    return turns[last_user]["content"], turns[:last_user]


class Assistant:
    def __init__(self, kb, client=None, top_k: int | None = None) -> None:
        self.kb = kb
        self.client = client
        self.retriever = Retriever(kb, client)
        self.top_k = top_k or env_int("TOP_K", 8, 1, 30)
        self.dense_ready = False

    @classmethod
    def from_path(cls, kb_path, client=None) -> "Assistant":
        return cls(load_kb(kb_path), client or LLMClient())

    def build_index(self) -> None:
        self.dense_ready = self.retriever.build_dense()

    def answer(self, question: str, history: list[dict] | None = None, allow_clarify: bool = False) -> dict:
        t0 = time.time()
        if history is None:
            question, history = parse_dialog(question)
        user_context = " ".join(t["content"] for t in history if t["role"] == "user")
        context = user_context
        last_assistant = next((t["content"] for t in reversed(history) if t["role"] == "assistant"), "")
        if last_assistant:
            context = f"{context} {last_assistant[:400]}"
        hits = self.retriever.search(question, top_k=self.top_k, context=context, entity_context=user_context)
        chunks = [h.chunk for h in hits]
        result = gen.generate(self.client, self.kb, question, history, chunks, allow_clarify)
        result["steps"] = []
        result["debug"] = {
            "retrieved": [(h.chunk.chunk_id, round(h.score, 4), h.sources) for h in hits],
            "seconds": round(time.time() - t0, 2),
        }
        return result
