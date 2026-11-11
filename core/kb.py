# Запуск для ручной проверки:
# python core/kb.py data/knowledge-base.zip --dump out/chunks.jsonl

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Iterator, Optional

# try:
#     import docx as _docx
# except Exception:
#     _docx = None


# ============ Константы ============

# "Своя" организация. Не хардкодим по файлу: одно место — одна замена.
SELF_ORG_NAME = "меридиан"

# Порог длины единицы закона, при котором дробим её на подпункты.
SUBCHUNK_THRESHOLD = 1200

# Жёсткий лимит длины чанка.
MAX_CHUNK_LEN = 2500

# Разрешённые расширения во входном архиве/каталоге.
ALLOWED_EXTS = {".txt", ".md", ".docx"}


# Латиница → кириллица для нормализации номеров договоров.
_LAT_TO_CYR_NUM = {
    'a': 'а', 'b': 'в', 'c': 'с', 'e': 'е', 'h': 'н',
    'k': 'к', 'm': 'м', 'o': 'о', 'p': 'р', 't': 'т',
    'x': 'х', 'y': 'у',
}


def normalize_number(s: str) -> str:
    """Единая нормализация номера договора.

    - нижний регистр;
    - латиница → кириллица для похожих букв (a→а, p→р, t→т, ...);
    - убираем '/', '-', '.', пробелы.
    """
    if not s:
        return ""
    s = s.lower()
    out = []
    for ch in s:
        if ch in _LAT_TO_CYR_NUM:
            out.append(_LAT_TO_CYR_NUM[ch])
        elif ch in ' -./':
            continue
        else:
            out.append(ch)
    return ''.join(out)


# ============ Dataclasses ============
class ContractsDict(dict):
    """Словарь договоров с нормализацией ключей при доступе."""

    def __getitem__(self, key):
        return super().__getitem__(normalize_number(key))

    def get(self, key, default=None):
        return super().get(normalize_number(key), default)

    def __contains__(self, key):
        return super().__contains__(normalize_number(key))

    def __setitem__(self, key, value):
        super().__setitem__(normalize_number(key), value)


@dataclass
class Document:
    doc_id: str
    doc_type: str
    title: str
    short_ref: str
    number: Optional[str] = None
    date: Optional[date] = None
    counterparty: Optional[dict] = None
    law_code: Optional[str] = None
    article: Optional[str] = None
    edition: Optional[str] = None
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None
    status: str = "active"
    amends: Optional[dict] = None
    addenda: list[str] = field(default_factory=list)
    other_editions: list[str] = field(default_factory=list)
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    exported_at: Optional[date] = None
    text: str = ""
    filename: str = ""

    def meta_snapshot(self) -> dict:
        return {
            "doc_type": self.doc_type,
            "number": self.number,
            "counterparty": self.counterparty["name"] if self.counterparty else None,
            "edition": self.edition,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "status": self.status,
        }


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ref: str
    section: Optional[str]
    text: str
    text_for_index: str
    start: int
    end: int
    repealed: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class Entities:
    counterparties: list[dict] = field(default_factory=list)
    contracts: dict[str, str] = field(default_factory=dict)
    documents_by_type: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class KnowledgeBase:
    documents: dict[str, Document]
    chunks: list[Chunk]
    entities: Entities
    kb_hash: str
    _chunk_by_id: dict = field(default_factory=dict, repr=False)
    _ref_index: dict = field(default_factory=dict, repr=False)

    def get_document(self, doc_id: str) -> Document:
        return self.documents[doc_id]

    def get_chunk(self, chunk_id: str) -> Chunk:
        if chunk_id in self._chunk_by_id:
            return self._chunk_by_id[chunk_id]
        raise KeyError(chunk_id)

    def find_ref(self, doc_id: str, ref: str) -> Optional[Chunk]:
        lst = self._ref_index.get((doc_id, _normalize_ref(ref)))
        return lst[0] if lst else None

    def find_ref_all(self, doc_id: str, ref: str) -> list[Chunk]:
        return list(self._ref_index.get((doc_id, _normalize_ref(ref)), []))

    def resolve_contract(self, text: str) -> Optional[str]:
        if not text:
            return None

        # 1) Кандидаты: сначала номер, выуженный регуляркой из текста,
        #    затем — весь текст целиком (для случаев, когда номер пришёл
        #    без обвязки).
        candidates: list[str] = []
        m = re.search(r'[A-Za-zА-Яа-я]{1,3}\s*-?\s*\d{1,4}\s*[/-]\s*\d{2,4}', text)
        if m:
            candidates.append(m.group(0))
        candidates.append(text)

        # 2) Для каждого кандидата: прямое попадание, затем попытка
        #    латинская «p» → кириллическая «п» в нормализованном номере.
        #    Замена р→п применяется к номеру, а не ко всему вопросу.
        for cand in candidates:
            doc_id = self.entities.contracts.get(cand)
            if doc_id:
                return doc_id
            n = normalize_number(cand)
            if n:
                alt = n.replace('р', 'п')
                if alt != n:
                    doc_id = self.entities.contracts.get(alt)
                    if doc_id:
                        return doc_id
        return None

    
    def match_counterparties(self, text: str) -> list[dict]:
        if not text:
            return []
        lower = text.lower()
        all_aliases: list[tuple[str, dict]] = []
        for cp in self.entities.counterparties:
            for alias in cp["aliases"]:
                all_aliases.append((alias.lower(), cp))
        all_aliases.sort(key=lambda x: -len(x[0]))

        consumed = [False] * len(lower)
        seen: set[str] = set()
        result: list[dict] = []
        for alias, cp in all_aliases:
            start = 0
            while True:
                idx = lower.find(alias, start)
                if idx < 0:
                    break
                end = idx + len(alias)
                left_ok = idx == 0 or not lower[idx - 1].isalnum()
                right_ok = end == len(lower) or not lower[end].isalnum()
                if left_ok and right_ok and not any(consumed[idx:end]):
                    if cp["name"] not in seen:
                        seen.add(cp["name"])
                        result.append(cp)
                    for k in range(idx, end):
                        consumed[k] = True
                    break
                start = idx + 1
        return result


# ============ Text utilities ============

_WS_RE = re.compile(r"[ \t\u00A0\u202F]+")

_TRANSLIT = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
}

_MONTHS = {
    'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5, 'июня': 6,
    'июля': 7, 'августа': 8, 'сентября': 9, 'октября': 10, 'ноября': 11,
    'декабря': 12,
}


def normalize_text(s: str) -> str:
    if not s:
        return ""
    if s.startswith("\ufeff"):
        s = s[1:]
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00A0", " ").replace("\u202F", " ")
    lines = [_WS_RE.sub(" ", ln).lstrip().rstrip() for ln in s.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def slugify(s: str) -> str:
    s = s.lower()
    out = []
    for ch in s:
        if ch in _TRANSLIT:
            out.append(_TRANSLIT[ch])
        elif ch.isalnum():
            out.append(ch)
        else:
            out.append('-')
    return re.sub(r'-+', '-', ''.join(out)).strip('-')


def parse_russian_date(s: str) -> Optional[date]:
    if not s:
        return None
    s = s.strip().lower()
    m = re.search(r'(\d{1,2})\s+([а-яё]+)\s+(\d{4})', s)
    if m and m.group(2) in _MONTHS:
        try:
            return date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))
        except ValueError:
            return None
    m = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', s)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    return None


def _normalize_ref(ref: str) -> str:
    r = ref.strip().lower()
    r = re.sub(r'^пункт\s+', 'п. ', r)
    r = re.sub(r'^часть\s+', 'ч. ', r)
    r = re.sub(r'^подпункт\s+', 'пп. ', r)
    r = re.sub(r'^примечание\s+', 'прим. ', r)
    r = re.sub(r'([а-яa-z]+\.)\s*(\d)', r'\1 \2', r)
    r = re.sub(r'\s+', ' ', r).strip()
    return r

def _strip_leading_number(text: str) -> str:
    # Дефисные номера ("1.7-1.", "2.8-1)") не срезаем: они уже есть в ref,
    # а частичная обрезка ("1." → оставляет "7-1.") ломает text_for_index.
    # Оставляем полный токен — BM25/эмбеддинги это переживут.
    if re.match(r'^\d+(?:\.\d+)*-\d+[\.\)]', text):
        return text
    return re.sub(r'^\d+(?:\.\d+)*[\.\)]\s*', '', text, count=1)

# ============ Rules: чистка законов от пометок Гаранта ============

GARANT_HEADER_RE = [
    re.compile(r'^Источник:\s*(.+)$', re.MULTILINE),
    re.compile(r'^URL:\s*(\S+)$', re.MULTILINE),
    re.compile(r'^Дата выгрузки:\s*(.+)$', re.MULTILINE),
]

# GARANT_HISTORY_LINE_RE = re.compile(
#     r'^\s*('
#     r'(?:Статья|Пункт|Подпункт|Часть|Абзац|Положени\w+)\s+'
#     r'(?:[\d\.]+\s+)?'
#     r'(?:изменен[аоы]?|дополнен[аоы]?|утратил[аи]?\s+силу|исключен[аоы]?)'
#     r'(?:\s+с\s+[\d\.]+\s*(?:г\.|года)?)?'
#     r'.*'
#     r'|Федеральн(?:ым|ыми|ый|ые)\s+закон\w*\s+от\s+.*'
#     r'|См\.\s+текст\s+.*'
#     r'|Информация\s+об\s+изменениях.*'
#     r'|О\s+проверке\s+конституционности.*'
#     r'|О\s+[^\n]{5,300}?\sсм\.\s+(?:постановлени|определени|письм|разъяснени)\w*.*'
#     # Строка «Cрок давности …» с латинской C (навигация Гаранта)
#     r'|\u0043рок\s+давност[ьи]\s+привлечения.*'
#     r')\s*$',
#     re.MULTILINE | re.IGNORECASE,
# )

GARANT_KEEP_RE = re.compile(
    r'Положения\s+настоящего\s+(?:подпункта|пункта|абзаца)'
    r'|^\s*\d+\)\s+утратил\s+силу',
)

GARANT_HISTORY_RES = [
    re.compile(
        r'^\s*(?:Статья|Пункт|Подпункт|Часть|Абзац|Положени\w+|Примечани[еяй])\s+'
        r'(?:[\d\.\-]+\s+)?'
        r'(?:изменен[аоы]?|дополнен[аоы]?|утратил[аи]?\s+силу|исключен[аоы]?)'
        r'(?:\s+с\s+[\d\.]+\s*(?:г\.|года)?)?'
        r'.*$',
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(r'^\s*Федеральн(?:ым|ыми|ый|ые)\s+закон\w*\s+от\s+.*$', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^\s*См\.\s+текст\s+.*$', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^\s*Информация\s+об\s+изменениях.*$', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^\s*О\s+проверке\s+конституционности.*$', re.MULTILINE | re.IGNORECASE),
    re.compile(
        r'^\s*О\s+[^\n]{5,300}?\sсм\.\s+(?:постановлени|определени|письм|разъяснени)\w*.*$',
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r'^\s*Обратите\s+внимание\b.*$',
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(r'^\s*\u0043рок\s+давност[ьи]\s+привлечения.*$', re.MULTILINE | re.IGNORECASE),
]


def clean_law_text(raw: str) -> tuple[str, dict]:
    header: dict = {}

    m = GARANT_HEADER_RE[0].search(raw)
    if m:
        header['source'] = m.group(1).strip()
    m = GARANT_HEADER_RE[1].search(raw)
    if m:
        header['source_url'] = m.group(1).strip()
    m = GARANT_HEADER_RE[2].search(raw)
    if m:
        header['exported_at_raw'] = m.group(1).strip()
        d = parse_russian_date(m.group(1))
        if d:
            header['exported_at'] = d

    cleaned_lines: list[str] = []
    for line in raw.split('\n'):
        if any(r.search(line) for r in GARANT_HEADER_RE):
            continue
        if any(r.search(line) for r in GARANT_HISTORY_RES) and not GARANT_KEEP_RE.search(line):
            continue
        cleaned_lines.append(line)

    out: list[str] = []
    blank = 0
    for ln in cleaned_lines:
        if not ln.strip():
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(ln)

    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()

    return "\n".join(out), header


# ============ Чтение архива ============

def _decode(filename: str, raw: bytes) -> str:
    name = filename.lower()
    if name.endswith(".docx"):
        return _read_docx(raw)

    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        text = raw.decode("cp1251")
        print(f"[kb] WARN: {filename} decoded as cp1251", file=sys.stderr)
        return text
    except UnicodeDecodeError:
        print(f"[kb] WARN: {filename} decoded with errors=replace", file=sys.stderr)
        return raw.decode("utf-8-sig", errors="replace")


_W_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


# def _read_docx(raw: bytes) -> str:
#     # Если python-docx не справился (неполный OPC-архив, битый файл и т.п.),
#     # откатываемся на ET, а не роняем разбор всего документа.
#     if _docx is not None:
#         try:
#             doc = _docx.Document(io.BytesIO(raw))
#             return "\n".join(p.text for p in doc.paragraphs)
#         except Exception as e:
#             print(
#                 f"[kb] WARN: python-docx не смог разобрать DOCX ({e}); "
#                 f"fallback на ElementTree",
#                 file=sys.stderr,
#             )
#             # не return — уходим в ET-ветку ниже

#     # Fallback через ElementTree — корректно раскодирует все XML-сущности,
#     # включая числовые (&#171; → «, &#187; → »).
#     with zipfile.ZipFile(io.BytesIO(raw)) as zf:
#         xml_bytes = zf.read("word/document.xml")
#     try:
#         root = ET.fromstring(xml_bytes)
#     except ET.ParseError:
#         return ""
#     paragraphs: list[str] = []
#     for p in root.iter(_W_NS + 'p'):
#         parts = [t.text or "" for t in p.iter(_W_NS + 't')]
#         paragraphs.append("".join(parts))
#     return "\n".join(paragraphs)

def _read_docx(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        xml_bytes = zf.read("word/document.xml")
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""
    paragraphs: list[str] = []
    for p in root.iter(_W_NS + 'p'):
        parts = [t.text or "" for t in p.iter(_W_NS + 't')]
        paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


def _iter_archive(path: Path) -> Iterator[tuple[str, bytes]]:
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if "__MACOSX/" in name or name.startswith("__MACOSX"):
                continue
            if info.file_size == 0:
                continue
            ext = Path(name).suffix.lower()
            if ext not in ALLOWED_EXTS:
                print(f"[kb] WARN: skip non-document {name}", file=sys.stderr)
                continue
            yield name, zf.read(info)


def _iter_dir(path: Path) -> Iterator[tuple[str, bytes]]:
    for f in path.iterdir():
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTS:
            yield f.name, f.read_bytes()


# ============ Классификация документа ============

def _first_significant_lines(text: str, n: int = 5) -> list[str]:
    """Первые n непустых строк, пропуская статусные строки «РЕДАКЦИЯ УТРАТИЛА СИЛУ…»."""
    result: list[str] = []
    for line in text.split('\n'):
        s = line.strip()
        if not s:
            continue
        if re.match(r'^РЕДАКЦИЯ\s+УТРАТИЛА\s+СИЛУ', s, re.IGNORECASE):
            continue
        result.append(s)
        if len(result) >= n:
            break
    return result


def extract_doc_type(filename: str, text: str) -> str:
    fn = filename.lower()

    # Законы — по уникальным маркерам
    if (re.search(r'^Источник:', text[:500], re.MULTILINE)
            or re.search(r'^URL:\s*https?://', text[:500], re.MULTILINE | re.IGNORECASE)
            or "__законы__" in fn or fn.startswith("законы__") or "/законы__" in fn):
        return "law"
    if re.search(r'^Статья\s+\d+(?:\.\d+)?\.', text[:3000], re.MULTILINE):
        return "law"

    # Тип — по первой значимой строке. Проверяем строки в порядке
    # следования: как только строка подошла под анкоренный шаблон — это тип.
    # Так договор, упоминающий «дополнительное соглашение» в преамбуле,
    # остаётся договором (первая строка «ДОГОВОР…» выигрывает).
    sig_lines = _first_significant_lines(text, 5)

    patterns = [
        ("letter",     re.compile(r'^\s*(?:письмо\b|исх\.\s*№)', re.IGNORECASE)),
        ("memo",       re.compile(r'^\s*служебная\s+записка\b', re.IGNORECASE)),
        ("addendum",   re.compile(r'^\s*(?:дополнительное|доп\.)\s+соглашение\b', re.IGNORECASE)),
        ("regulation", re.compile(r'^\s*регламент\b', re.IGNORECASE)),
        ("contract",   re.compile(r'^\s*договор\b', re.IGNORECASE)),
    ]
    for line in sig_lines:
        for doc_type, pat in patterns:
            if pat.search(line):
                return doc_type

    # Fallback по имени файла — только когда текст ничего не дал
    if "письмо" in fn:
        return "letter"
    if "служебная записка" in fn:
        return "memo"
    if "дополнительное соглашение" in fn or "доп. соглашение" in fn or "доп соглашение" in fn:
        return "addendum"
    if "регламент" in fn:
        return "regulation"
    if "договор" in fn:
        return "contract"

    return "other"

# ============ Извлечение метаданных ============

_ORG_RE = re.compile(
    r'(ООО|АО|ПАО|ЗАО|ИП)\s*[«"]([^»"]+)[»"]\s*\(?ИНН\s*(\d{9,12})',
    re.IGNORECASE,
)
_INN_RE = re.compile(r'ИНН\s*(\d{9,12})')
_NUMBER_RE = re.compile(r'№\s*([А-ЯA-Z0-9\-/]+)', re.IGNORECASE)


def extract_counterparty(text: str, doc_type: str) -> Optional[dict]:
    if doc_type not in ("contract", "addendum"):
        return None

    head = text[:1500]
    candidates: list[dict] = []
    for m in _ORG_RE.finditer(head):
        name = f"{m.group(1)} «{m.group(2)}»"
        candidates.append({"name": name, "inn": m.group(3)})

    # Без ИНН сразу после названия: «ООО «X» («Поставщик»)» / «ООО «X», именуемое…»
    for m in re.finditer(
        r'(ООО|АО|ПАО|ЗАО|ИП)\s*[«"]([^»"]+)[»"]\s*(?:\([^)]*\)|,)?',
        head,
        re.IGNORECASE,
    ):
        name = f"{m.group(1)} «{m.group(2)}»"
        if not any(c["name"] == name for c in candidates):
            candidates.append({"name": name, "inn": None})

    if not candidates:
        print(f"[kb] WARN: no counterparty in contract preamble", file=sys.stderr)
        return None

    for c in candidates:
        if SELF_ORG_NAME not in c["name"].lower():
            return c

    # Все найденные — наша организация; контрагента нет
    print(f"[kb] WARN: only self org in contract preamble", file=sys.stderr)
    return None


def extract_number(text: str, doc_type: str) -> Optional[str]:
    if doc_type == "law":
        return None

    if doc_type == "regulation":
        # Ищем номер собственного приказа, а не того, что отменил редакцию
        m = re.search(
            r'Утверждён\w*\s+(?:приказом|распоряжением)\b[^№\n]*№\s*([А-ЯA-Z0-9\-/]+)',
            text[:1500],
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()

    head = text[:800]
    m = _NUMBER_RE.search(head)
    if m:
        return m.group(1).strip()
    return None


def extract_title(filename: str, text: str, doc_type: str) -> str:
    if doc_type == "law":
        m = re.search(r'^Статья\s+[\d\.]+\..*$', text, re.MULTILINE)
        if m:
            return m.group(0).strip()
        return filename

    if doc_type == "regulation":
        for line in text.split('\n'):
            s = line.strip()
            if not s:
                continue
            if re.match(r'^РЕДАКЦИЯ\s+УТРАТИЛА\s+СИЛУ', s, re.IGNORECASE):
                continue
            if re.match(r'^(ООО|АО|ПАО|ЗАО)\b', s):
                continue
            if re.match(r'^Утвержд', s, re.IGNORECASE):
                continue
            if 'регламент' in s.lower():
                # Убираем «(утратил силу)» и подобное
                s = re.sub(r'\s*[\(\[]\s*утратил\w*\s+силу\s*[\)\]]?', '', s, flags=re.IGNORECASE)
                return s.strip()
        base = filename.split("__", 1)[-1]
        base = re.sub(r'\.(txt|md|docx)$', '', base, flags=re.IGNORECASE)
        base = re.sub(r'[\s_,]*[\(\[]?\s*ред\.?\s*\d{4}\s*[\)\]]?', '', base, flags=re.IGNORECASE).strip(" _,")
        return base or filename

    if doc_type == "contract":
        first = text.split("\n", 1)[0].strip() if text else ""
        if first:
            head_part, sep, tail = first.partition("№")
            hp = head_part.strip()
            if hp and (hp.isupper() or hp[:1].isupper()):
                hp = hp.lower().capitalize()
            if sep:
                return f"{hp} №{tail}"
            return hp or first

    if doc_type in ("memo", "letter"):
        # §3.3: title — тема документа («О …» / «Об …», «Служебная записка …»,
        # «Письмо …»), а не «Исх. № …» и не «ООО «…»».
        # Два прохода: сначала тематические строки, потом «Исх. № …» как фолбэк.
        for line in text.split('\n'):
            s = line.strip()
            if not s:
                continue
            if re.match(r'^(ООО|АО|ПАО|ЗАО)\b', s):
                continue
            if s.lower().startswith(('о ', 'об ')):
                return s
            if 'служебная записка' in s.lower() or s.lower().startswith('письмо'):
                return s
        for line in text.split('\n'):
            s = line.strip()
            if re.match(r'^Исх\.\s*№', s, re.IGNORECASE):
                return s
        first = text.split("\n", 1)[0].strip() if text else ""
        if first:
            return first

    first = text.split("\n", 1)[0].strip() if text else ""
    if first:
        return first
    base = filename.split("__", 1)[-1]
    return base.rsplit(".", 1)[0]


_LAW_CODES = [
    (r'(?<![а-яё])(?:нк|налогов\w+\s+кодекс)\b', "НК РФ"),
    (r'(?<![а-яё])(?:тк|трудов\w+\s+кодекс)\b', "ТК РФ"),
    (r'(?<![а-яё])(?:коап|кодекс\s+об\s+административн\w+)\b', "КоАП РФ"),
]


def _detect_law_code(src: str, allow_fz: bool = True) -> Optional[str]:
    s = src.lower()
    for pat, code in _LAW_CODES:
        if re.search(pat, s):
            return code
    if allow_fz:
        m = re.search(r'(?<!\d)(\d{2,3})\s*[-–]?\s*фз\b', s)
        if m:
            return f"{m.group(1)}-ФЗ"
    if "мрот" in s:
        return "Закон о МРОТ"
    m = re.search(r'\b([А-ЯЁ]{2,6})\s+РФ\b', src)
    if m:
        return f"{m.group(1)} РФ"
    return None


def extract_law_code_and_article(filename: str, text: str,
                                 source: Optional[str] = None
                                 ) -> tuple[Optional[str], Optional[str]]:
    art_pat = re.compile(r'(?:ст\.?\s*|Статья\s+)(\d+(?:\.\d+)?)')
    body_art = re.search(r'^Статья\s+(\d+(?:\.\d+)?)\.', text, re.MULTILINE)

    for src in (filename, source):
        if not src:
            continue
        code = _detect_law_code(src)
        if code:
            am = art_pat.search(src) or body_art
            return code, (am.group(1) if am else None)

    code = _detect_law_code(text[:3000], allow_fz=False)
    am = body_art or art_pat.search(text[:3000])
    return code, (am.group(1) if am else None)


def extract_edition(text: str) -> Optional[str]:
    m = re.search(r'Редакция\s+от\s+(\d{2}\.\d{2}\.\d{4})', text)
    if not m:
        m = re.search(r'Редакция\s+от\s+(\d{1,2}\s+[а-яё]+\s+\d{4})', text)
    return f"от {m.group(1)}" if m else None


def extract_validity(text: str, doc_type: str = "other"
                     ) -> tuple[Optional[date], Optional[date], str]:
    vf = vt = None
    status = "active"

    m = re.search(r'Действует\s+с\s+(\d{2}\.\d{2}\.\d{4}|\d{1,2}\s+[а-яё]+\s+\d{4})',
                  text, re.IGNORECASE)
    if m:
        vf = parse_russian_date(m.group(1))

    m = re.search(r'УТРАТИЛА\s+СИЛУ\s+с\s+(\d{2}\.\d{2}\.\d{4}|\d{1,2}\s+[а-яё]+\s+\d{4})',
                  text, re.IGNORECASE)
    if m:
        vt = parse_russian_date(m.group(1))
        if vt:
            status = "expired"

    if vf is None:
        m = re.search(r'Редакция\s+от\s+(\d{2}\.\d{2}\.\d{4}|\d{1,2}\s+[а-яё]+\s+\d{4})', text)
        if m:
            vf = parse_russian_date(m.group(1))

    # NEW: письма вида «С 21 апреля 2025 г. до подписания …»
    if vf is None and doc_type == "letter":
        m = re.search(
            r'^\s*[Сс]\s+(\d{1,2}\s+[а-яё]+\s+\d{4})\s+г?\.?\s+'
            r'(?:до\s+подписания|вступает\s+в\s+силу|применяется)',
            text[:1500],
            re.MULTILINE | re.IGNORECASE,
        )
        if m:
            vf = parse_russian_date(m.group(1))

    return vf, vt, status

def extract_amends(text: str) -> Optional[dict]:
    head = text[:1500]
    parent_number = None
    m = re.search(r'СОГЛАШЕНИЕ\s+№\s*(\S+)\s+к\s+договору\s+[^№]*№\s*(\S+)',
                  head, re.IGNORECASE)
    if m:
        parent_number = m.group(2).rstrip(".,;")
    else:
        m = re.search(r'к\s+договору\s+[^№]*№\s*(\S+)', head, re.IGNORECASE)
        if m:
            parent_number = m.group(1).rstrip(".,;")

    points: list[str] = []
    for pm in re.finditer(r'пункт[аы]?\s+(\d+\.\d+(?:\s*,\s*\d+\.\d+)*)', text, re.IGNORECASE):
        for num in re.findall(r'\d+\.\d+', pm.group(1)):
            if num not in points:
                points.append(num)
    for pm in re.finditer(r'\bп\.\s*(\d+\.\d+)', text):
        if pm.group(1) not in points:
            points.append(pm.group(1))

    condition = None
    cm = re.search(r'Условие\s+применения[:\s]*[«"]?(.+?)[»"]?(?:\.\s|\.$|$)', text,
                   re.IGNORECASE | re.DOTALL)
    if cm:
        condition = cm.group(1).strip()
        tail = text[cm.end(1):cm.end(1) + 3]
        if re.match(r'[»"]?\s*\.', tail) and not condition.endswith('.'):
            condition += '.'
    else:
        cm = re.search(r'применя\w*\s+(к\s+.+?)(?:\.\s|\.$)', text, re.IGNORECASE)
        if cm:
            condition = cm.group(1).strip()

    return {"parent_number": parent_number, "points": points, "condition": condition}


def build_doc_id(doc_type, filename, title, number, law_code, article,
                 edition, parent_number: Optional[str] = None) -> str:
    if doc_type == "law" and law_code and article:
        code_key = slugify(law_code)
        code_key = code_key.replace('-rf', '').replace('-fz', 'fz')
        if code_key.startswith("zakon-o-"):
            code_key = code_key[len("zakon-o-"):]
        return f"{code_key}_{article.replace('.', '-')}"

    if doc_type == "contract" and number:
        return f"contract_{slugify(number)}"

    if doc_type == "addendum":
        num_match = re.search(r'№\s*(\S+)', title)
        num_part = slugify(num_match.group(1)) if num_match else 'x'
        if parent_number:
            return f"addendum_{num_part}_{slugify(parent_number)}"
        return f"addendum_{num_part}"

    if doc_type == "regulation":
        words = [w for w in re.split(r'\s+', title) if w and w[0].isalpha()]
        if words and slugify(words[0]).startswith("reglament"):
            words = words[1:]
        key = slugify(words[0]) if words else "doc"
        year = ""
        if edition:
            ym = re.search(r'(\d{4})', edition)
            if ym:
                year = f"_{ym.group(1)}"
        return f"reglament_{key}{year}"

    if doc_type == "memo" and number:
        return f"memo_{slugify(number)}"
    if doc_type == "letter" and number:
        return f"letter_{slugify(number)}"

    return f"{doc_type}_{slugify(title)[:40] or 'doc'}"


# ============ Парсинг одного документа ============

def parse_document(filename: str, raw_text: str) -> Document:
    raw = normalize_text(raw_text)
    doc_type = extract_doc_type(filename, raw)

    if doc_type == "law":
        cleaned, header = clean_law_text(raw)
    else:
        cleaned, header = raw, {}

    title = extract_title(filename, cleaned, doc_type)
    number = extract_number(cleaned, doc_type)
    law_code, article = (
        extract_law_code_and_article(filename, cleaned, header.get("source"))
        if doc_type == "law" else (None, None)
    )

    edition = extract_edition(cleaned)
    if doc_type == "law":
        edition = "действующая"

    if doc_type == "law":
        valid_from, valid_to, status = None, None, "active"
    else:
        valid_from, valid_to, status = extract_validity(cleaned, doc_type)

    counterparty = extract_counterparty(cleaned, doc_type)

    doc_date = None
    if doc_type in ("contract", "addendum"):
        head = cleaned[:500]
        dm = re.search(r'[—\-–]\s*([\d]{1,2}\s+[а-яё]+\s+\d{4})', head, re.IGNORECASE)
        if dm:
            doc_date = parse_russian_date(dm.group(1))
        if not doc_date:
            dm = re.search(r'от\s+([\d]{2}\.[\d]{2}\.[\d]{4})', head)
            if dm:
                doc_date = parse_russian_date(dm.group(1))
    if doc_type in ("memo", "letter"):
        dm = re.search(r'от\s+([\d]{2}\.[\d]{2}\.[\d]{4})', cleaned[:300])
        if dm:
            doc_date = parse_russian_date(dm.group(1))

    if doc_type == "contract" and valid_from is None:
        valid_from = doc_date

    amends = None
    parent_number = None
    if doc_type == "addendum":
        parsed = extract_amends(cleaned)
        if parsed:
            parent_number = parsed.get("parent_number")
            amends = {
                "points": parsed.get("points", []),
                "condition": parsed.get("condition"),
            }

    short_ref = _make_short_ref(doc_type, title, number, counterparty,
                                law_code, article, edition, parent_number)
    doc_id = build_doc_id(doc_type, filename, title, number,
                          law_code, article, edition,
                          parent_number=parent_number)

    if doc_type in ("contract", "addendum") and valid_to is None:
        m = re.search(
            r'действует\s+до\s+(\d{1,2}\s+[а-яё]+\s+\d{4}|\d{2}\.\d{2}\.\d{4})',
            cleaned, re.IGNORECASE,
        )
        if m:
            valid_to = parse_russian_date(m.group(1))

    return Document(
        doc_id=doc_id,
        doc_type=doc_type,
        title=title,
        short_ref=short_ref,
        number=number,
        date=doc_date,
        counterparty=counterparty,
        law_code=law_code,
        article=article,
        edition=edition,
        valid_from=valid_from,
        valid_to=valid_to,
        status=status,
        amends=amends,
        source_url=header.get("source_url"),
        source_title=header.get("source"),
        exported_at=header.get("exported_at"),
        text=cleaned,
        filename=filename,
    )


def _make_short_ref(doc_type, title, number, counterparty, law_code, article,
                    edition, parent_number: Optional[str] = None) -> str:
    if doc_type == "law" and law_code and article:
        return f"{law_code}, ст. {article}"
    if doc_type == "contract" and number:
        kind_part = title.split("№", 1)[0].strip()
        words = kind_part.split()
        kind = " ".join(words[:2]).lower() if len(words) >= 2 else kind_part.lower()
        return f"{kind} № {number}"
    if doc_type == "addendum" and number:
        if parent_number:
            return f"доп. соглашение № {number} к договору № {parent_number}"
        return f"доп. соглашение № {number}"
    if doc_type == "memo" and number:
        return f"служебная записка № {number}"
    if doc_type == "letter" and number:
        return f"письмо № {number}"
    if doc_type == "regulation":
        base = re.sub(r'\s+', ' ', title).strip()
        return f"{base}, редакция {edition}" if edition else base
    return title[:80]


# ============ Chunking ============

_BLANK_LINE_RE = re.compile(r"\n\s*\n")


def iter_paragraphs(text: str) -> Iterator[tuple[str, int, int]]:
    pos, n = 0, len(text)
    while pos < n:
        m = _BLANK_LINE_RE.search(text, pos)
        raw_end = m.start() if m else n
        s, e = pos, raw_end
        while s < e and text[s] in " \n":
            s += 1
        while e > s and text[e - 1] in " \n":
            e -= 1
        if e > s:
            yield text[s:e], s, e
        pos = m.end() if m else n


def _status_display(doc: Document) -> str:
    if doc.status == "expired":
        verb = "утратил силу" if doc.doc_type in ("contract", "addendum") else "утратила силу"
        if doc.valid_to:
            return f"{verb} с {doc.valid_to.strftime('%d.%m.%Y')}"
        return verb
    if doc.doc_type in ("contract", "addendum"):
        return "действующий"
    return "действующая"


def _doc_prefix_for_index(doc: Document) -> str:
    base = doc.short_ref
    if doc.doc_type == "contract" and doc.counterparty:
        base = f"{base} с {doc.counterparty['name']}"
    if base and base[0].islower():
        base = base[0].upper() + base[1:]
    return f"{base} ({_status_display(doc)})."

def _ref_for_index(ref: str) -> str:
    if not ref:
        return ref
    if ref[0].islower() and not ref.startswith(("п.", "ч.", "пп.", "прим.")):
        return ref[0].upper() + ref[1:]
    return ref


def _build_text_for_index(doc: Document, ref: str, section: Optional[str], content: str) -> str:
    parts = [_doc_prefix_for_index(doc)]
    if section and not ref.startswith("раздел"):
        parts.append(f"Раздел {section}.")
    parts.append(_ref_for_index(ref) + ".")
    parts.append(content)
    return " ".join(parts)

_REPEALED_RE = re.compile(
    r'^\W*(?:\d+(?:\.\d+)*(?:-\d+)?[.)]\s*)?'
    r'(?:(?:Часть|Пункт|Подпункт|Абзац)\s+[\w.\-]+\s+)?утратил[аи]?\s+силу',
    re.IGNORECASE,
)


def _make_chunk(doc: Document, slug: str, ref: str, section: Optional[str],
                text: str, start: int, end: int) -> Chunk:
    content = _strip_leading_number(text)
    tfid = _build_text_for_index(doc, ref, section, content)
    meta = doc.meta_snapshot()
    if _is_suspicious(text):
        meta["suspicious"] = True
    return Chunk(
        chunk_id=f"{doc.doc_id}#{slug}",
        doc_id=doc.doc_id,
        ref=ref,
        section=section,
        text=text,
        text_for_index=tfid,
        start=start,
        end=end,
        repealed=bool(_REPEALED_RE.match(text.strip())),
        meta=meta,
    )


_SUSPICIOUS_RE = re.compile(
    r'(игнорируй|забудь\s+(?:все\s+)?инструкции|для\s+ии\b|'
    r'system\s*prompt|отвечай\s+(?:только|всегда)|как\s+ии\b|'
    r'ты\s*[-—]\s*ассистент|ignore\s+(?:all\s+)?previous\s+instructions)',
    re.IGNORECASE,
)


def _is_suspicious(text: str) -> bool:
    return bool(_SUSPICIOUS_RE.search(text))


# --- Договоры и регламенты ---

_SECTION_RE = re.compile(r'^(\d+)\.\s+([А-ЯЁA-Z][^\n]*)$', re.MULTILINE)
_POINT_RE = re.compile(r'^(\d+\.\d+)\.\s+', re.MULTILINE)


def _chunk_contract_like(doc: Document) -> list[Chunk]:
    text = doc.text
    markers = []
    for m in _SECTION_RE.finditer(text):
        markers.append(("section", m.start(), m.group(1), m.group(2).strip(), None))
    for m in _POINT_RE.finditer(text):
        markers.append(("point", m.start(), None, None, m.group(1)))
    markers.sort(key=lambda x: x[1])

    chunks: list[Chunk] = []
    first = markers[0][1] if markers else len(text)
    if first > 0:
        pre_text = text[:first].strip()
        if pre_text:
            s, e = 0, first
            while s < e and text[s] in " \n":
                s += 1
            while e > s and text[e - 1] in " \n":
                e -= 1
            chunks.append(_make_chunk(doc, "preamble", "преамбула", None, text[s:e], s, e))

    current_section: Optional[str] = None
    i = 0
    while i < len(markers):
        kind, pos, sec_num, sec_title, point_num = markers[i]
        if kind == "section":
            current_section = f"{sec_num}. {sec_title}"
            has_points = False
            for j in range(i + 1, len(markers)):
                if markers[j][0] == "section":
                    break
                if markers[j][0] == "point":
                    has_points = True
                    break
            if not has_points:
                next_pos = markers[i + 1][1] if i + 1 < len(markers) else len(text)
                s, e = pos, next_pos
                while e > s and text[e - 1] in " \n":
                    e -= 1
                slug = "requisites" if "реквизит" in current_section.lower() else f"s{sec_num}"
                ref = f"раздел {sec_num} (реквизиты)" if slug == "requisites" else f"раздел {sec_num}"
                chunks.append(_make_chunk(doc, slug, ref, current_section, text[s:e], s, e))
        else:
            next_pos = markers[i + 1][1] if i + 1 < len(markers) else len(text)
            s, e = pos, next_pos
            while e > s and text[e - 1] in " \n":
                e -= 1
            slug = f"p{point_num}"
            ref = f"п. {point_num}"
            chunks.append(_make_chunk(doc, slug, ref, current_section, text[s:e], s, e))
        i += 1

    return _split_oversized(chunks, doc=doc)

def _split_oversized(chunks: list[Chunk], max_len: int = MAX_CHUNK_LEN,
                     doc: Optional[Document] = None) -> list[Chunk]:
    out: list[Chunk] = []
    for c in chunks:
        if len(c.text) <= max_len:
            out.append(c)
            continue

        pieces: list[Chunk] = []
        par_pieces = list(iter_paragraphs(c.text))
        if not par_pieces or max(len(p[0]) for p in par_pieces) > max_len:
            pieces = _hard_split(c, max_len, doc=doc)
        else:
            buf_s, buf_e = -1, -1
            idx = 0
            for _t, ps, pe in par_pieces:
                if buf_s < 0:
                    buf_s, buf_e = ps, pe
                elif pe - buf_s <= max_len:
                    buf_e = pe
                else:
                    piece = c.text[buf_s:buf_e]
                    pieces.append(_clone_with_text(c, piece, c.start + buf_s, c.start + buf_e, idx, doc=doc))
                    idx += 1
                    buf_s, buf_e = ps, pe
            if buf_s >= 0:
                piece = c.text[buf_s:buf_e]
                pieces.append(_clone_with_text(c, piece, c.start + buf_s, c.start + buf_e, idx, doc=doc))

        if len(pieces) > 1:
            for k, p in enumerate(pieces):
                p.meta["part_index"] = k
                p.meta["part_count"] = len(pieces)
        out.extend(pieces)
    return out

_END_PUNCT_RE = re.compile(r'[\.\!\?\;\:]')


def _hard_split(c: Chunk, max_len: int, doc: Optional[Document] = None) -> list[Chunk]:
    text = c.text
    n = len(text)
    if n == 0:
        return [c]

    boundaries = sorted({m.end() for m in _END_PUNCT_RE.finditer(text)})
    if not boundaries or boundaries[-1] != n:
        boundaries.append(n)

    out: list[Chunk] = []
    idx = 0
    start = 0
    while start < n:
        end = start
        for b in boundaries:
            if b <= start:
                continue
            if b - start <= max_len:
                end = b
            else:
                break
        if end == start:
            space = text.rfind(' ', start, start + max_len)
            end = space if space > start else start + max_len
        real_end = end
        while real_end > start and text[real_end - 1].isspace():
            real_end -= 1
        piece = text[start:real_end]
        if piece:
            out.append(_clone_with_text(
                c, piece,
                start=c.start + start,
                end=c.start + real_end,
                idx=idx,
                doc=doc,
            ))
            idx += 1
        start = end
        while start < n and text[start].isspace():
            start += 1

    return out


def _clone_with_text(src: Chunk, text: str, start: int, end: int, idx: int,
                     doc: Optional[Document] = None) -> Chunk:
    slug = src.chunk_id.split("#", 1)[1] + (f"_{idx}" if idx else "")
    tfid = (
        _build_text_for_index(doc, src.ref, src.section, _strip_leading_number(text))
        if doc else text
    )
    return Chunk(
        chunk_id=f"{src.doc_id}#{slug}",
        doc_id=src.doc_id,
        ref=src.ref,
        section=src.section,
        text=text,
        text_for_index=tfid,
        start=start,
        end=end,
        repealed=src.repealed,
        meta=dict(src.meta),
    )


# --- Законы ---

def _law_unit_names(code: Optional[str]) -> tuple[str, Optional[str]]:
    code = code or ""
    if "НК" in code:
        return "п", "пп"
    if "КоАП" in code:
        return "ч", None
    if "152" in code:
        return "ч", "п"
    if "-ФЗ" in code:
        return "ч", "п"
    return "ч", None


_IMPERATIVE_RE = re.compile(
    r'^\s*(?:Установить|Определить|Предоставить|Обеспечить|Ввести|'
    r'Признать|Утвердить|Возложить|Применять|Применяется|'
    r'Установлен\w*|Определен\w*|Введен\w*|Дополнить|Исключить|Отменить)\b',
    re.IGNORECASE,
)


def _is_norm_article_line(rest: str, source: Optional[str] = None) -> bool:
    s = rest.strip()
    if not s:
        return False
    if source:
        s_norm = re.sub(r'\s+', ' ', s.lower())
        src_norm = re.sub(r'\s+', ' ', source.lower())
        if s_norm in src_norm:
            return False
        return True
    if len(s) > 200 and re.search(r'[\.;]$', s):
        return True
    if _IMPERATIVE_RE.match(s):
        return True
    return False

def _chunk_law_numbered(doc: Document, source: Optional[str] = None) -> list[Chunk]:
    text = doc.text
    top_name, sub_name = _law_unit_names(doc.law_code)

    body_start = 0
    art_m = re.search(r'^Статья\s+([\d\.]+)\.\s*(.*)$', text, re.MULTILINE)
    if art_m:
        body_start = art_m.end()

    notes_match = re.search(
        r'^\s*Примечани[ея]\b.*$',
        text[body_start:],
        re.MULTILINE | re.IGNORECASE,
    )
    notes_pos = body_start + notes_match.start() if notes_match else None

    paras = [(t, s, e) for t, s, e in iter_paragraphs(text) if s >= body_start]
    if not paras:
        return _fallback_chunks(doc)

    top_re = re.compile(r'^(\d+(?:\.\d+)*(?:-\d+)?)\.\s')
    tops = [(i, m.group(1), s, e)
            for i, (t, s, e) in enumerate(paras)
            if (m := top_re.match(t))]
    if not tops:
        return _fallback_chunks(doc)

    # P0-1: откат нумерации — только при реальном уменьшении целой части
    rollback_pos = None
    prev = 0
    for _, num, s, _ in tops:
        try:
            n = int(num.split('.')[0])
        except ValueError:
            prev = 0
            continue
        if n < prev and prev >= 3:
            rollback_pos = s
            break
        prev = n
    if rollback_pos is not None and (notes_pos is None or rollback_pos < notes_pos):
        notes_pos = rollback_pos

    chunks: list[Chunk] = []

    # Переходный текст до «Статья N.»
    pre_paras = [(t, s, e) for t, s, e in iter_paragraphs(text) if e <= (art_m.start() if art_m else 0)]
    if pre_paras and art_m:
        pre_s = pre_paras[0][1]
        pre_e = pre_paras[-1][2]
        while pre_e > pre_s and text[pre_e - 1] in " \n":
            pre_e -= 1
        if pre_e > pre_s:
            chunks.append(_make_chunk(
                doc, "trans_pre",
                f"переходное положение к ст. {doc.article}" if doc.article else "переходное положение",
                None, text[pre_s:pre_e], pre_s, pre_e,
            ))

    # P0-3: текст между строкой «Статья N.» и первым пунктом
    if tops:
        first_top_start = tops[0][2]
        mid_paras = [(t, s, e) for t, s, e in iter_paragraphs(text)
                     if body_start <= s and e <= first_top_start]
        if mid_paras:
            mid_s = mid_paras[0][1]
            mid_e = mid_paras[-1][2]
            while mid_e > mid_s and text[mid_e - 1] in " \n":
                mid_e -= 1
            if mid_e > mid_s:
                chunks.append(_make_chunk(
                    doc, "trans_body",
                    f"переходное положение к ст. {doc.article}" if doc.article else "переходное положение",
                    None, text[mid_s:mid_e], mid_s, mid_e,
                ))

    seen_refs: set[str] = set()

    def _emit(slug: str, ref: str, s: int, e: int) -> None:
        if ref in seen_refs:
            print(f"[kb] WARN: duplicate ref {ref!r} in {doc.doc_id}", file=sys.stderr)
            slug = f"{slug}_dup"
        seen_refs.add(ref)
        chunks.append(_make_chunk(doc, slug, ref, None, text[s:e], s, e))

    # P1-4: заголовок примечаний, содержащий номер на той же строке
    notes_header_has_number = False
    if notes_pos is not None:
        line_end = text.find('\n', notes_pos)
        if line_end < 0:
            line_end = len(text)
        first_line = text[notes_pos:line_end]
        m = re.match(r'^\s*Примечани[ея][.:]?\s*(\d+(?:\.\d+)?)\.\s', first_line)
        if m:
            notes_header_has_number = True
            num = m.group(1)
            next_top_start = len(text)
            for _, _, ts, _ in tops:
                if ts > notes_pos:
                    next_top_start = ts
                    break
            note_end = next_top_start
            while note_end > notes_pos and text[note_end - 1] in " \n":
                note_end -= 1
            _emit(f"note{num}",
                  f"прим. {num} к ст. {doc.article}" if doc.article else f"прим. {num}",
                  notes_pos, note_end)

    first_note_chunk_done = notes_header_has_number

    for ti, (pi, num, ps, pe) in enumerate(tops):
        end_of_top = tops[ti + 1][2] if ti + 1 < len(tops) else len(text)
        if notes_pos is not None and ps < notes_pos < end_of_top:
            end_of_top = notes_pos
        end_of_top = _trim_right(text, ps, end_of_top)
        in_notes = notes_pos is not None and ps >= notes_pos

        slug_base = f"note{num}" if in_notes else f"p{num}"
        if in_notes:
            ref_base = (f"прим. {num} к ст. {doc.article}"
                        if doc.article else f"прим. {num}")
        else:
            ref_base = (f"{top_name}. {num} ст. {doc.article}"
                        if doc.article else f"{top_name}. {num}")

        actual_start = ps
        if in_notes and not first_note_chunk_done and notes_pos is not None:
            actual_start = notes_pos
            first_note_chunk_done = True

        subs: list[tuple[str, int, int]] = []
        if sub_name and not in_notes:
            range_text = text[actual_start:end_of_top]
            sub_matches = list(re.finditer(
                r'^(\d+(?:\.\d+)*(?:-\d+)?)\)\s', range_text, re.MULTILINE,
            ))
            for k, sm in enumerate(sub_matches):
                sub_num = sm.group(1)
                sub_start = actual_start + sm.start()
                sub_end = (actual_start + sub_matches[k + 1].start()
                           if k + 1 < len(sub_matches) else end_of_top)
                while sub_end > sub_start and text[sub_end - 1] in " \n":
                    sub_end -= 1
                subs.append((sub_num, sub_start, sub_end))

        top_len = end_of_top - actual_start
        has_subs = bool(subs)
        if not has_subs or top_len <= SUBCHUNK_THRESHOLD:
            _emit(slug_base, ref_base, actual_start, end_of_top)
            continue

        intro_end = _trim_right(text, actual_start, subs[0][1])
        if intro_end > actual_start:
            _emit(slug_base, ref_base, actual_start, intro_end)

        for si, (subnum, ss, se) in enumerate(subs):
            sub_slug = f"{slug_base}.pp{subnum}"
            sub_ref = f"{sub_name}. {subnum} {ref_base}"
            _emit(sub_slug, sub_ref, ss, se)

    return _split_oversized(chunks, doc=doc)


_NUM_ITEM_RE = re.compile(r'^\d+(?:\.\d+)?\)\s')
_LAW_NOTE_RE = re.compile(
    r'^Положени\w+\s+(?:статьи|части|пункта)\s+[\d\.]+\s*\(в\s+редакции',
    re.IGNORECASE,
)


def _chunk_law_paragraphs(doc: Document, source: Optional[str] = None) -> list[Chunk]:
    text = doc.text
    art_m = re.search(r'^Статья\s+([\d\.]+)\.\s*(.*)$', text, re.MULTILINE)
    if not art_m:
        return _fallback_chunks(doc)

    article_num = art_m.group(1)
    rest = art_m.group(2)
    body_start = art_m.end()

    chunks: list[Chunk] = []
    part_counter = 0

    # Переходный текст до «Статья N.»
    pre_paras = [(t, s, e) for t, s, e in iter_paragraphs(text) if e <= art_m.start()]
    if pre_paras:
        pre_s = pre_paras[0][1]
        pre_e = pre_paras[-1][2]
        while pre_e > pre_s and text[pre_e - 1] in " \n":
            pre_e -= 1
        if pre_e > pre_s:
            chunks.append(_make_chunk(
                doc, "trans_pre", f"переходное положение к ст. {article_num}",
                None, text[pre_s:pre_e], pre_s, pre_e,
            ))

    # Норма в строке заголовка → это первая часть статьи
    if _is_norm_article_line(rest, source=source):
        norm_start = art_m.end() - len(rest)
        line_end = art_m.end()
        while line_end > norm_start and text[line_end - 1] in " \n":
            line_end -= 1
        if line_end > norm_start:
            part_counter += 1
            chunks.append(_make_chunk(
                doc, f"ch{part_counter}", f"ч. {part_counter} ст. {article_num}",
                None, text[norm_start:line_end], norm_start, line_end,
            ))

    paras = [(t, s, e) for t, s, e in iter_paragraphs(text) if s >= body_start]
    if not paras:
        return chunks if chunks else _fallback_chunks(doc)

    groups: list[tuple[int, int, bool]] = []
    cur = None
    prev_colon = prev_open = False
    for t, s, e in paras:
        tt = t.rstrip()
        is_note = bool(_LAW_NOTE_RE.match(t))
        item = bool(_NUM_ITEM_RE.match(t)) or t[:1].islower()
        cont = (cur is not None and not is_note and not cur[2] and (
            prev_colon
            or (prev_open and item)
            or (t[:1].islower() and tt.endswith((";", ".")))
        ))
        if cont:
            cur[1] = e
        else:
            if cur:
                groups.append(tuple(cur))
            cur = [s, e, is_note]
        prev_colon = tt.endswith(":")
        prev_open = tt.endswith((":", ";"))
    if cur:
        groups.append(tuple(cur))

    n_note = 0
    for s, e, is_note in groups:
        if is_note:
            n_note += 1
            slug = f"trans{n_note}"
            ref = (f"переходное положение к ст. {article_num}"
                   if article_num else f"переходное положение {n_note}")
        else:
            part_counter += 1
            slug = f"ch{part_counter}"
            ref = f"ч. {part_counter} ст. {article_num}" if article_num else f"ч. {part_counter}"
        chunks.append(_make_chunk(doc, slug, ref, None, text[s:e], s, e))

    return _split_oversized(chunks, doc=doc)


def _chunk_law(doc: Document, source: Optional[str] = None) -> list[Chunk]:
    has_numbering = bool(re.search(r'^\d+(?:\.\d+)?\.\s', doc.text, re.MULTILINE))
    if has_numbering:
        return _chunk_law_numbered(doc, source=source)
    return _chunk_law_paragraphs(doc, source=source)



def _trim_right(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1] in " \n":
        end -= 1
    return end


# --- Допсоглашения/записки/письма ---

def _chunk_whole(doc: Document) -> list[Chunk]:
    if not doc.text.strip():
        return []
    s, e = 0, len(doc.text)
    while s < e and doc.text[s] in " \n":
        s += 1
    while e > s and doc.text[e - 1] in " \n":
        e -= 1

    if doc.doc_type == "addendum":
        slug = "whole"
        ref = f"доп. соглашение № {doc.number}" if doc.number else "доп. соглашение"
    elif doc.doc_type == "memo":
        slug = "whole"
        ref = f"служебная записка № {doc.number}" if doc.number else "служебная записка"
    elif doc.doc_type == "letter":
        slug = "whole"
        ref = f"письмо № {doc.number}" if doc.number else "письмо"
    else:
        slug, ref = "whole", doc.short_ref

    chunks = [_make_chunk(doc, slug, ref, None, doc.text[s:e], s, e)]
    return _split_oversized(chunks, doc=doc)


def _chunk_memo_letter(doc: Document) -> list[Chunk]:
    if len(doc.text) < 2000:
        return _chunk_whole(doc)
    chunks: list[Chunk] = []
    for idx, (t, s, e) in enumerate(iter_paragraphs(doc.text), start=1):
        chunks.append(_make_chunk(doc, f"par{idx}", f"абзац {idx}", None, t, s, e))
    return _split_oversized(chunks, doc=doc)


def _fallback_chunks(doc: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, (t, s, e) in enumerate(iter_paragraphs(doc.text), start=1):
        chunks.append(_make_chunk(doc, f"p{idx}", doc.short_ref, None, t, s, e))
    if not chunks:
        return _chunk_whole(doc)
    return _split_oversized(chunks, doc=doc)


def chunk_document(doc: Document) -> list[Chunk]:
    if doc.doc_type in ("contract", "regulation"):
        return _chunk_contract_like(doc)
    if doc.doc_type == "law":
        return _chunk_law(doc, source=doc.source_title)
    if doc.doc_type in ("addendum", "memo", "letter", "other"):
        if doc.doc_type in ("memo", "letter"):
            return _chunk_memo_letter(doc)
        return _chunk_whole(doc)
    return _fallback_chunks(doc)


# Связи между документами 

def _link_addenda(documents: dict[str, Document]) -> None:
    contract_by_number: dict[str, str] = {}
    for d in documents.values():
        if d.doc_type == "contract" and d.number:
            contract_by_number[normalize_number(d.number)] = d.doc_id

    for d in documents.values():
        if d.doc_type != "addendum" or not d.amends:
            continue
        parsed = extract_amends(d.text)
        if not parsed:
            continue
        parent_num = parsed.get("parent_number")
        parent_id = None
        if parent_num:
            parent_id = contract_by_number.get(normalize_number(parent_num))

        d.amends = {
            "doc_id": parent_id,
            "points": parsed.get("points", []),
            "condition": parsed.get("condition"),
        }

        if parent_id:
            parent = documents[parent_id]
            if d.doc_id not in parent.addenda:
                parent.addenda.append(d.doc_id)
            if d.counterparty is None or not d.counterparty.get("inn"):
                d.counterparty = parent.counterparty
            if d.number:
                parent_ref_instr = re.sub(r'\bдоговор\b', 'договору', parent.short_ref, count=1)
                d.short_ref = f"доп. соглашение № {d.number} к {parent_ref_instr}"
            else:
                parent_ref_instr = re.sub(r'\bдоговор\b', 'договору', parent.short_ref, count=1)
                d.short_ref = f"доп. соглашение к {parent_ref_instr}"


def _link_regulations(documents: dict[str, Document]) -> None:
    regs = [d for d in documents.values() if d.doc_type == "regulation"]
    for a in regs:
        wa = a.title.lower().split()[:3]
        if not wa:
            continue
        for b in regs:
            if a is b:
                continue
            wb = b.title.lower().split()[:3]
            if wb and wa == wb:
                if b.doc_id not in a.other_editions:
                    a.other_editions.append(b.doc_id)


def _annotate_amendments(documents: dict[str, Document], chunks: list[Chunk]) -> None:
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    for add_doc in documents.values():
        if add_doc.doc_type != "addendum" or not add_doc.amends:
            continue
        parent_id = add_doc.amends.get("doc_id")
        points = add_doc.amends.get("points") or []
        if not parent_id:
            continue
        for point in points:
            slug = f"p{point}"
            target_id = f"{parent_id}#{slug}"
            for c in by_doc.get(parent_id, []):
                if c.chunk_id == target_id:
                    lst = c.meta.setdefault("amended_by", [])
                    if add_doc.doc_id not in lst:
                        lst.append(add_doc.doc_id)


# ============ Entities ============

def build_entities(documents: dict[str, Document]) -> Entities:
    counterparties: list[dict] = []
    # contracts: dict[str, str] = {}
    contracts: ContractsDict = ContractsDict()
    by_type: dict[str, list[str]] = {}

    for d in documents.values():
        by_type.setdefault(d.doc_type, []).append(d.doc_id)

        if d.doc_type == "contract" and d.number:
            key = normalize_number(d.number)
            if key:
                contracts[key] = d.doc_id

        if d.counterparty:
            name = d.counterparty["name"]
            short = re.sub(r'^(ООО|АО|ПАО|ЗАО|ИП)\s*[«"]', '', name).rstrip('»"')
            alias_variants = {name, short, short.replace("-", " "), short.replace(" ", "-")}
            aliases = {a for a in alias_variants if a}
            for ex in counterparties:
                if ex["name"] == name:
                    if d.doc_id not in ex["doc_ids"]:
                        ex["doc_ids"].append(d.doc_id)
                    ex["aliases"] = sorted(set(ex["aliases"]) | aliases)
                    break
            else:
                counterparties.append({
                    "name": name,
                    "aliases": sorted(aliases),
                    "inn": d.counterparty.get("inn"),
                    "doc_ids": [d.doc_id],
                })

    return Entities(
        counterparties=counterparties,
        contracts=contracts,
        documents_by_type=by_type,
    )


# ============ editions_on ============

def editions_on(d: date, docs: list[Document]) -> list[Document]:
    out: list[Document] = []
    for doc in docs:
        if doc.valid_from and doc.valid_from > d:
            continue
        if doc.valid_to and doc.valid_to <= d:
            continue
        out.append(doc)
    return out


# ============ load_kb ============

def load_kb(path: str | Path) -> KnowledgeBase:
    path = Path(path)

    if path.is_dir():
        files = sorted(_iter_dir(path), key=lambda x: x[0])
    else:
        files = sorted(_iter_archive(path), key=lambda x: x[0])

    h = hashlib.sha256()
    for name, data in files:
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(data)
    kb_hash = h.hexdigest()

    # 1. Документы
    documents: dict[str, Document] = {}
    for filename, raw in files:
        try:
            raw_text = _decode(filename, raw)
        except Exception as e:
            print(f"[kb] WARN: cannot decode {filename}: {e}", file=sys.stderr)
            continue
        try:
            doc = parse_document(filename, raw_text)
        except Exception as e:
            print(f"[kb] WARN: cannot parse {filename}: {e}", file=sys.stderr)
            doc = Document(
                doc_id=f"other_{slugify(filename)[:40]}",
                doc_type="other", title=filename, short_ref=filename,
                text=normalize_text(raw_text), filename=filename,
            )
        if doc.doc_id in documents:
            i = 2
            while f"{doc.doc_id}_{i}" in documents:
                i += 1
            doc.doc_id = f"{doc.doc_id}_{i}"
        documents[doc.doc_id] = doc

    # 1.5. Связи до чанкинга — чтобы short_ref допсоглашения уже был с родителем
    _link_addenda(documents)
    _link_regulations(documents)

    # 2. Чанки
    chunks: list[Chunk] = []
    for doc in documents.values():
        try:
            chunks.extend(chunk_document(doc))
        except Exception as e:
            print(f"[kb] WARN: cannot chunk {doc.doc_id}: {e}", file=sys.stderr)
            chunks.extend(_fallback_chunks(doc))

    # safety-net по уникальности chunk_id
    seen_ids: dict[str, int] = {}
    for c in chunks:
        if c.chunk_id in seen_ids:
            seen_ids[c.chunk_id] += 1
            c.chunk_id = f"{c.chunk_id}_dup{seen_ids[c.chunk_id]}"
            c.meta["duplicate_ref"] = True
        else:
            seen_ids[c.chunk_id] = 1
    if any(c.meta.get("duplicate_ref") for c in chunks):
        n = sum(1 for c in chunks if c.meta.get("duplicate_ref"))
        print(f"[kb] WARN: {n} chunk_id collisions auto-renamed", file=sys.stderr)

    _annotate_amendments(documents, chunks)
    entities = build_entities(documents)

    # Индексы
    chunk_by_id: dict[str, Chunk] = {}
    ref_index: dict[tuple[str, str], list[Chunk]] = {}
    for c in chunks:
        chunk_by_id[c.chunk_id] = c
        ref_index.setdefault((c.doc_id, _normalize_ref(c.ref)), []).append(c)

    return KnowledgeBase(
        documents=documents,
        chunks=chunks,
        entities=entities,
        kb_hash=kb_hash,
        _chunk_by_id=chunk_by_id,
        _ref_index=ref_index,
    )


# ============ CLI ============

def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kb.py", description="Knowledge base loader")
    ap.add_argument("path", help="zip-архив или каталог с базой")
    ap.add_argument("--dump", default=None, help="куда записать chunks.jsonl")
    ap.add_argument("--pretty", action="store_true", help="Отступы для читаемости")
    args = ap.parse_args(argv)

    try:
        kb = load_kb(args.path)
    except Exception as e:
        print(f"[kb] ERROR: {e}", file=sys.stderr)
        return 1

    print(f"documents={len(kb.documents)} chunks={len(kb.chunks)} "
          f"contracts={len(kb.entities.contracts)}", file=sys.stderr)

    if args.dump:
        out = Path(args.dump)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for c in kb.chunks:
                if args.pretty:
                    f.write(json.dumps(asdict(c), ensure_ascii=False, indent=2) + "\n")
                else:
                    f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        print(f"dumped {len(kb.chunks)} chunks → {out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())