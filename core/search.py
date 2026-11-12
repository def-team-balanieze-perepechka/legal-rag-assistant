from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from stem import normalize, stem, tokenize

RRF_K = 60
CANDIDATES = 30          # Сколько возьмем из каждого ретривера
RERANK_POOL = 24         # Сколько кандидатов отдаем реранкеру
EMBED_BATCH = 32
PINNED_DOC_MAX_CHUNKS = 25  # Документ подкладываем целиком, если он не длиннее этого

_LAT2CYR = str.maketrans("abcehkmoptxy", "авсенкмортху")
_NUMBER_RE = re.compile(r"(?<![\w])([a-zа-я]{1,3})\s*-?\s*(\d{1,4})\s*[/-]\s*(\d{2,4})(?![\w])", re.I)


_QUERY_TAX = (
    (re.compile(r"\bНДС\b|добавленн\w*\s+стоимост", re.I), "НДС, налог на добавленную стоимость"),
    (re.compile(r"\bНДФЛ\b|доход\w*\s+физических\s+лиц", re.I), "НДФЛ, налог на доходы физических лиц"),
    (re.compile(r"налог\w*\s+на\s+прибыль", re.I), "налог на прибыль организаций"),
)


def query_tax_topic(text: str) -> str | None:
    """
    Вид налога, названный в вопросе
    (в тех же названиях, что kb.tax_topic); несколько — None.
    """
    found = {name for rx, name in _QUERY_TAX if rx.search(text or "")}
    return found.pop() if len(found) == 1 else None


def number_key(s: str) -> str:
    """
    Номер договора в каноническом виде:
    нижний регистр, кириллица, без разделителей.
    """
    return re.sub(r"[\s\-/.]", "", normalize(s).translate(_LAT2CYR))


@dataclass
class Hit:
    chunk: object
    score: float
    sources: list[str] = field(default_factory=list)


class EntityMatcher:
    """Находит в тексте номера договоров и названия контрагентов из справочника базы."""

    def __init__(self, kb) -> None:
        self.contracts = {number_key(k): v for k, v in kb.entities.contracts.items()}
        self.patterns: list[tuple[re.Pattern, list[str], str]] = []
        for cp in kb.entities.counterparties:
            names = set(cp.get("aliases") or []) | {cp["name"]}
            for alias in names:
                core = re.sub(r"^(ООО|АО|ПАО|ЗАО|ИП)\s*", "", alias).strip("«»\" ")
                words = re.findall(r"[а-яa-z0-9]+", normalize(core))
                if not words:
                    continue
                # Каждое слово - основа + короткое окончание;
                # дефис/пробел между словами взаимозаменяемы
                body = r"[\s-]+".join(re.escape(stem(w)) + r"[а-я]{0,4}" for w in words)
                self.patterns.append((re.compile(rf"(?<![а-яa-z0-9]){body}(?![а-яa-z0-9])"),
                                      list(cp["doc_ids"]), core))
        self.patterns.sort(key=lambda p: len(p[2]), reverse=True)

    def match(self, text: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {"contracts": [], "counterparties": []}
        for m in _NUMBER_RE.finditer(text):
            key = number_key("".join(m.groups()))
            # Латинская P похожа на «Р», но в номерах договоров обычно означает «П»
            doc_id = self.contracts.get(key) or self.contracts.get(key.replace("р", "п"))
            if doc_id and doc_id not in found["contracts"]:
                found["contracts"].append(doc_id)
        low = normalize(text)
        for pattern, doc_ids, _ in self.patterns:
            m = pattern.search(low)
            if m:
                low = low[: m.start()] + " " * (m.end() - m.start()) + low[m.end():]
                for d in doc_ids:
                    if d not in found["counterparties"]:
                        found["counterparties"].append(d)
        return found


class Retriever:
    def __init__(self, kb, client=None) -> None:
        self.kb = kb
        self.client = client
        self.chunks = list(kb.chunks)
        self.index_of = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self.by_doc: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            self.by_doc.setdefault(c.doc_id, []).append(i)
        self.bm25 = BM25Okapi([tokenize(c.text_for_index) or ["_"] for c in self.chunks])
        self.vectors: np.ndarray | None = None
        self.entities = EntityMatcher(kb)
        self.edition_groups = self._edition_groups()


    def build_dense(self) -> bool:
        if self.client is None:
            return False
        try:
            texts = [c.text_for_index for c in self.chunks]
            vecs: list[list[float]] = []
            for i in range(0, len(texts), EMBED_BATCH):
                vecs.extend(self.client.embed(texts[i:i + EMBED_BATCH]))
            arr = np.asarray(vecs, dtype=np.float32)
            self.vectors = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
            return True
        except Exception as exc:
            print(f"[search] эмбеддинги недоступны ({type(exc).__name__}: {exc}); работаем на BM25",
                  file=sys.stderr)
            self.vectors = None
            return False

    def _edition_groups(self) -> dict[str, list[str]]:
        """Редакции одного документа (регламенты): doc_id → doc_id других редакций."""
        groups: dict[str, list[str]] = {}
        docs = [d for d in self.kb.documents.values() if d.doc_type == "regulation"]
        for d in docs:
            key = normalize(re.sub(r",?\s*ред(?:\.|акция).*$", "", d.short_ref))
            other = [x.doc_id for x in docs if x is not d and
                     normalize(re.sub(r",?\s*ред(?:\.|акция).*$", "", x.short_ref)) == key]
            explicit = list(getattr(d, "other_editions", []) or [])
            groups[d.doc_id] = sorted(set(other + explicit))
        return groups


    def _bm25_top(self, query: str) -> list[int]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(-scores)[:CANDIDATES]
        return [int(i) for i in order if scores[i] > 0]

    def _dense_top(self, query: str) -> list[int]:
        if self.vectors is None or self.client is None:
            return []
        try:
            q = np.asarray(self.client.embed([query])[0], dtype=np.float32)
        except Exception as exc:
            print(f"[search] эмбеддинг запроса не получен: {type(exc).__name__}", file=sys.stderr)
            return []
        q /= max(float(np.linalg.norm(q)), 1e-12)
        sims = self.vectors @ q
        return [int(i) for i in np.argsort(-sims)[:CANDIDATES]]

    @staticmethod
    def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return scores

    def _rerank(self, query: str, pool: list[int]) -> list[tuple[int, float]] | None:
        if self.client is None or not pool:
            return None
        try:
            res = self.client.rerank(query, [self.chunks[i].text_for_index for i in pool], top_n=len(pool))
        except Exception as exc:
            print(f"[search] реранкер недоступен: {type(exc).__name__}", file=sys.stderr)
            return None
        return [(pool[i], s) for i, s in res if 0 <= i < len(pool)] or None

    def search(self, query: str, top_k: int = 8, context: str = "",
               entity_context: str | None = None) -> list[Hit]:
        """
        query — последняя реплика пользователя;
        context — предыдущие реплики диалога (для поиска);
        entity_context — реплики пользователя, по которым определяются договоры и контрагенты.
        Реплики ассистента часто перечисляют варианты («Альфа-Трейд или Альфа-Трейдинг?»),
        поэтому по ним сущности берутся, только если пользователь сам ничего не назвал.
        """
        full = f"{context}\n{query}".strip()
        bm25 = self._bm25_top(full)
        dense = self._dense_top(full)
        fused = self.rrf([bm25, dense])
        sources: dict[int, list[str]] = {}
        for name, ranking in (("bm25", bm25), ("dense", dense)):
            for i in ranking:
                sources.setdefault(i, []).append(name)

        # Документы, явно названные в вопросе
        ent = self.entities.match(f"{entity_context}\n{query}" if entity_context is not None else full)
        if entity_context is not None and not (ent["contracts"] or ent["counterparties"]):
            ent = self.entities.match(full)
        named_docs = ent["contracts"] or ent["counterparties"]
        pinned: list[int] = []
        for doc_id in named_docs:
            doc = self.kb.documents.get(doc_id)
            idxs = self.by_doc.get(doc_id, [])
            if doc is not None and doc.doc_type != "law" and len(idxs) <= PINNED_DOC_MAX_CHUNKS:
                pinned.extend(idxs)
            for add_id in getattr(doc, "addenda", []) or []:
                pinned.extend(self.by_doc.get(add_id, []))
        # Чанки других договоров с похожим контрагентом убираем,
        # если документ назван однозначно
        if named_docs:
            named = set(named_docs)
            for d in list(named):
                named.update(getattr(self.kb.documents.get(d), "addenda", []) or [])
            fused = {i: s for i, s in fused.items()
                     if self.chunks[i].doc_id in named
                     or self.kb.documents[self.chunks[i].doc_id].doc_type not in ("contract", "addendum")}

        # Назван вид налога — статьи о других налогах (тот же «Налоговые ставки»)
        # убираем по текущему вопросу; иначе — по прошлым репликам пользователя
        topic = query_tax_topic(query) or query_tax_topic(entity_context if entity_context is not None else context)
        if topic:
            fused = {i: s for i, s in fused.items()
                     if self.kb.documents[self.chunks[i].doc_id].topic in (None, topic)}

        pool = sorted(fused, key=fused.get, reverse=True)[:RERANK_POOL]
        reranked = self._rerank(query if not context else full, pool)
        if reranked:
            ranked = [(i, s) for i, s in sorted(reranked, key=lambda x: -x[1])]
            for i, _ in ranked:
                sources.setdefault(i, []).append("rerank")
        else:
            ranked = [(i, fused[i]) for i in pool]

        chosen: list[int] = [i for i, _ in ranked[:top_k]]
        score = dict(ranked)
        # Пункт с допсоглашением -> само допсоглашение;
        # пункт регламента -> тот же пункт в других редакциях
        for i in list(chosen):
            c = self.chunks[i]
            for add_id in c.meta.get("amended_by", []) or []:
                chosen.extend(self.by_doc.get(add_id, []))
            for other in self.edition_groups.get(c.doc_id, []):
                same = [j for j in self.by_doc.get(other, []) if self.chunks[j].ref == c.ref]
                for j in same:
                    sources.setdefault(j, []).append("edition")
                chosen.extend(same)
        for i in pinned:
            sources.setdefault(i, []).append("entity")
        result: list[int] = []
        for i in chosen + pinned:
            if i not in result:
                result.append(i)
        # Подкрепленные сущностью документы — в исходном порядке пунктов,
        # остальное — по релевантности
        return [Hit(self.chunks[i], float(score.get(i, 0.0)), sources.get(i, [])) for i in result]
