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
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass, field, asdict


# Опциональный python-docx 
try:
    import docx as _docx 
except Exception:  
    _docx = None


# Базовые классы
@dataclass
class Document:
    doc_id: str
    doc_type: str # law|contract|addendum|regulation|memo|letter|other
    title: str
    short_ref: str
    number: Optional[str] = None
    date: Optional[date] = None
    counterparty: Optional[dict] = None # {"name": str, "inn": str}
    law_code: Optional[str] = None
    article: Optional[str] = None
    edition: Optional[str] = None
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None
    status: str = "active"   # active|expired
    amends: Optional[dict] = None     # {"doc_id":..., "points":[...], "condition": str|None}
    addenda: list[str] = field(default_factory=list)
    other_editions: list[str] = field(default_factory=list)
    source_url: Optional[str] = None
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

    def get_document(self, doc_id: str) -> Document:
        return self.documents[doc_id]

    def get_chunk(self, chunk_id: str) -> Chunk:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        raise KeyError(chunk_id)

    def find_ref(self, doc_id: str, ref: str) -> Optional[Chunk]:
        target = _normalize_ref(ref)
        for c in self.chunks:
            if c.doc_id == doc_id and _normalize_ref(c.ref) == target:
                return c
        return None


# Text utilities
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
    'декабря': 12
}


def normalize_text(s: str) -> str:
    """Нормализуем переводы строк и NBSP, но не трогаем буквы."""
    if not s:
        return ""
    if s.startswith("\ufeff"):
        s = s[1:]
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\u00A0", " ").replace("\u202F", " ")
    lines = [_WS_RE.sub(" ", ln).rstrip() for ln in s.split("\n")]
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
    # "1 августа 2024 г." / "1 августа 2024"
    m = re.search(r'(\d{1,2})\s+([а-яё]+)\s+(\d{4})', s)
    if m and m.group(2) in _MONTHS:
        try:
            return date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))
        except ValueError:
            return None
    # "12.05.2025"
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
    # "ч.4" → "ч. 4", но "3.1" не трогаем
    r = re.sub(r'([а-яa-z]+\.)\s*(\d)', r'\1 \2', r)
    r = re.sub(r'\s+', ' ', r).strip()
    return r


def _strip_leading_number(text: str) -> str:
    """Убираем '3.1. ' / '2) ' / '1. ' в начале — чтобы не дублировать номер в text_for_index."""
    return re.sub(r'^\d+(?:\.\d+)*[\.\)]\s*', '', text, count=1)



# Rules: чистка законов от пометок Гаранта
GARANT_HEADER_RE = [
    re.compile(r'^Источник:\s*(.+)$', re.MULTILINE),
    re.compile(r'^URL:\s*(\S+)$', re.MULTILINE),
    re.compile(r'^Дата выгрузки:\s*(.+)$', re.MULTILINE),
]

GARANT_HISTORY_LINE_RE = re.compile(
    r'^\s*('
    r'(?:Статья|Пункт|Подпункт|Часть|Абзац|Положени\w+)\s+'
    r'(?:[\d\.]+\s+)?'
    r'(?:изменен[аоы]?|дополнен[аоы]?|утратил[аи]?\s+силу|исключен[аоы]?)'
    r'(?:\s+с\s+[\d\.]+\s*(?:г\.|года)?)?'
    r'.*'
    r'|Федеральн(?:ым|ыми|ый|ые)\s+закон\w*\s+от\s+.*'
    r'|См\.\s+текст\s+.*'
    r'|Информация\s+об\s+изменениях.*'
    r'|О\s+проверке\s+конституционности.*'
    r'|О\s+[^\n]{5,300}?\sсм\.\s+(?:постановлени|определени|письм|разъяснени)\w*.*'
    r')\s*$',
    re.MULTILINE,
)
# Нормы, а не пометки
GARANT_KEEP_RE = re.compile(
    r'Положения\s+настоящего\s+(?:подпункта|пункта|абзаца)'
    r'|^\s*\d+\)\s+утратил\s+силу',
)


def clean_law_text(raw: str) -> tuple[str, dict]:
    """Возвращает (очищенный текст, шапка)."""
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
        if GARANT_HISTORY_LINE_RE.match(line) and not GARANT_KEEP_RE.search(line):
            continue
        cleaned_lines.append(line)

    # Схлопываем лишние пустые строки
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



# Чтение архива
def _decode(filename: str, raw: bytes) -> str:
    name = filename.lower()
    if name.endswith(".docx"):
        return _read_docx(raw)
    # txt / md — декодируем utf-8-sig
    return raw.decode("utf-8-sig", errors="replace")


def _read_docx(raw: bytes) -> str:
    if _docx is not None:
        doc = _docx.Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)
    # Fallback: читаем word/document.xml напрямую
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    xml = re.sub(r'</w:p>', '\n', xml)
    xml = re.sub(r'<[^>]+>', '', xml)
    # Простейший unescape
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'")):
        xml = xml.replace(a, b)
    return xml


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
            yield name, zf.read(info)


def _iter_dir(path: Path) -> Iterator[tuple[str, bytes]]:
    for f in path.iterdir():
        if f.is_file() and f.suffix.lower() in {".txt", ".md", ".docx"}:
            yield f.name, f.read_bytes()



# Классификация документа
def extract_doc_type(filename: str, text: str) -> str:
    fn = filename.lower()
    head = text[:800].lower()
    body2k = text[:2000].lower()

    # Законы
    if re.search(r'^Источник:', text[:500], re.MULTILINE) \
       or re.search(r'^URL:\s*https?://', text[:500], re.MULTILINE | re.IGNORECASE) \
       or "__законы__" in fn or fn.startswith("законы__") or "/законы__" in fn:
        return "law"
    if re.search(r'^Статья\s+\d+(?:\.\d+)?\.', text[:3000], re.MULTILINE):
        return "law"

    # Письмо/записка — по filename и по НАЧАЛУ строки, не по вхождению слова в тело.
    # Письмо может цитировать регламент — поэтому идёт раньше regulation.
    if "письмо" in fn \
       or re.search(r'^\s*письмо\b', text[:400], re.IGNORECASE) \
       or re.search(r'^\s*исх\.\s*№', text[:500], re.MULTILINE | re.IGNORECASE):
        return "letter"
    if "служебная записка" in fn or "служебная записка" in head:
        return "memo"
    if ("дополнительное соглашение" in fn or "доп. соглашение" in fn
        or "дополнительное соглашение" in head or "доп. соглашение" in head):
        return "addendum"
    # Регламент — только filename или заголовок-первая-строка, не «упоминание»
    if "регламент" in fn or re.search(r'^\s*регламент\b', text[:400], re.IGNORECASE):
        return "regulation"
    if re.search(r'^\s*договор\b', text[:300], re.IGNORECASE) or "договор" in fn:
        return "contract"

    return "other"



# Извлечение метаданных
_ORG_RE = re.compile(r'(ООО|АО|ПАО|ЗАО|ИП)\s*[«"]([^»"]+)[»"]\s*\(?ИНН\s*(\d{9,12})', re.IGNORECASE)
_INN_RE = re.compile(r'ИНН\s*(\d{9,12})')
_NUMBER_RE = re.compile(r'№\s*([А-ЯA-Z0-9\-/]+)', re.IGNORECASE)


def extract_counterparty(text: str, doc_type: str) -> Optional[dict]:
    if doc_type not in ("contract", "addendum"):
        return None
    # Ищем все организации в шапке; берём ту, что не «Меридиан»
    head = text[:1500]
    candidates = []
    for m in _ORG_RE.finditer(head):
        name = f"{m.group(1)} «{m.group(2)}»"
        candidates.append({"name": name, "inn": m.group(3)})
    for c in candidates:
        if "меридиан" not in c["name"].lower():
            return c
    return candidates[0] if candidates else None


def extract_number(text: str, doc_type: str) -> Optional[str]:
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

    if doc_type == "regulation":
        # Первая строка может быть «РЕДАКЦИЯ УТРАТИЛА СИЛУ …» — берём из имени
        base = filename.split("__", 1)[-1]
        base = re.sub(r'\.(txt|md|docx)$', '', base, flags=re.IGNORECASE)
        base = re.sub(r'[\s_,]*[\(\[]?\s*ред\.?\s*\d{4}\s*[\)\]]?',
                      '', base, flags=re.IGNORECASE).strip(" _,")
        if base:
            return base

    first = text.split("\n", 1)[0].strip() if text else ""
    if first:
        return first
    base = filename.split("__", 1)[-1]
    return base.rsplit(".", 1)[0]

_LAW_CODES = [
    (r'(?<![а-яё])(?:нк|налогов\w+\s+кодекс)\b',        "НК РФ"),
    (r'(?<![а-яё])(?:тк|трудов\w+\s+кодекс)\b',         "ТК РФ"),
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
    return None

def extract_law_code_and_article(filename: str, text: str,
                                 source: Optional[str] = None
                                 ) -> tuple[Optional[str], Optional[str]]:
    art_pat = re.compile(r'(?:ст\.?\s*|Статья\s+)(\d+(?:\.\d+)?)')
    body_art = re.search(r'^Статья\s+(\d+(?:\.\d+)?)\.', text, re.MULTILINE)

    # 1) имя файла, 2) строка «Источник:» из шапки Гаранта (она вырезана из text!)
    for src in (filename, source):
        if not src:
            continue
        code = _detect_law_code(src)
        if code:
            am = art_pat.search(src) or body_art
            return code, (am.group(1) if am else None)

    # 3) тело: только кодексы, без NN-ФЗ — там это ссылки на изменяющие законы
    code = _detect_law_code(text[:3000], allow_fz=False)
    am = body_art or art_pat.search(text[:3000])
    return code, (am.group(1) if am else None)


def extract_edition(text: str) -> Optional[str]:
    m = re.search(r'Редакция\s+от\s+(\d{2}\.\d{2}\.\d{4})', text)
    if not m:
        m = re.search(r'Редакция\s+от\s+(\d{1,2}\s+[а-яё]+\s+\d{4})', text)
    return f"от {m.group(1)}" if m else None


def extract_validity(text: str) -> tuple[Optional[date], Optional[date], str]:
    vf = vt = None
    status = "active"

    m = re.search(r'Действует\s+с\s+(\d{2}\.\d{2}\.\d{4}|\d{1,2}\s+[а-яё]+\s+\d{4})', text)
    if m:
        vf = parse_russian_date(m.group(1))

    m = re.search(r'УТРАТИЛА\s+СИЛУ\s+с\s+(\d{2}\.\d{2}\.\d{4}|\d{1,2}\s+[а-яё]+\s+\d{4})',
                  text, re.IGNORECASE)
    if m:
        vt = parse_russian_date(m.group(1))
        if vt:
            status = "expired"

    # valid_from для редакций — из «Редакция от …», независимо от vt
    if vf is None:
        m = re.search(r'Редакция\s+от\s+(\d{2}\.\d{2}\.\d{4}|\d{1,2}\s+[а-яё]+\s+\d{4})', text)
        if m:
            vf = parse_russian_date(m.group(1))

    return vf, vt, status


def extract_amends(text: str) -> Optional[dict]:
    """Для допсоглашений: parent_number, points, condition (без doc_id — разрешим позже)."""
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
    cm = re.search(r'Условие\s+применения[:\s]*[«"]?(.+?)[»"]?(?:\.\s|\.$|$)', text, re.IGNORECASE | re.DOTALL)
    if cm:
        condition = cm.group(1).strip()
    else:
        cm = re.search(r'применя\w*\s+(к\s+.+?)(?:\.\s|\.$)', text, re.IGNORECASE)
        if cm:
            condition = cm.group(1).strip()

    return {"parent_number": parent_number, "points": points, "condition": condition}


def build_doc_id(doc_type, filename, title, number, law_code, article,
                 edition, parent_number: Optional[str] = None) -> str:
    if doc_type == "law" and law_code and article:
        code_key = slugify(law_code)              # nk-rf, tk-rf, koap-rf, 152-fz, zakon-o-mrot
        code_key = code_key.replace('-rf', '').replace('-fz', 'fz')
        if code_key.startswith("zakon-o-"):
            code_key = code_key[len("zakon-o-"):]  # mrot
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

# Парсинг одного документа
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
    valid_from, valid_to, status = extract_validity(cleaned)
    counterparty = extract_counterparty(cleaned, doc_type)

    doc_date = None
    if doc_type == "contract":
        head = cleaned[:500]
        dm = re.search(r'[—\-–]\s*([\d]{1,2}\s+[а-яё]+\s+\d{4})', head, re.IGNORECASE)
        if dm:
            doc_date = parse_russian_date(dm.group(1))
        if not doc_date:
            dm = re.search(r'от\s+([\d]{2}\.[\d]{2}\.[\d]{4})', head)
            if dm:
                doc_date = parse_russian_date(dm.group(1))
    if doc_type == "memo" or doc_type == "letter":
        dm = re.search(r'от\s+([\d]{2}\.[\d]{2}\.[\d]{4})', cleaned[:300])
        if dm:
            doc_date = parse_russian_date(dm.group(1))

    if doc_type == "contract" and valid_from is None:
        valid_from = doc_date
    if doc_type == "law":
        valid_from = valid_from  # закон — действующая редакция; valid_from не критичен

     # допсоглашения: сначала parent_number, потом doc_id 
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
            # "doc_id" здесь не заполняем — его разрешит _link_addenda

    short_ref = _make_short_ref(doc_type, title, number, counterparty,
                                law_code, article, edition)

    doc_id = build_doc_id(doc_type, filename, title, number,
                          law_code, article, edition,
                          parent_number=parent_number)
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
        exported_at=header.get("exported_at"),
        text=cleaned,
        filename=filename,
    )


def _make_short_ref(doc_type, title, number, counterparty, law_code, article, edition) -> str:
    if doc_type == "law" and law_code and article:
        return f"{law_code}, ст. {article}"
    if doc_type == "contract" and number:
        kind_part = title.split("№", 1)[0].strip()
        words = kind_part.split()
        kind = " ".join(words[:2]).lower() if len(words) >= 2 else kind_part.lower()
        return f"{kind} № {number}"
    if doc_type == "addendum" and number:
        return f"доп. соглашение № {number}"
    if doc_type == "memo" and number:
        return f"служебная записка № {number}"
    if doc_type == "letter" and number:
        return f"письмо № {number}"
    if doc_type == "regulation":
        base = re.sub(r'\s+', ' ', title).strip()
        return f"{base}, редакция {edition}" if edition else base
    return title[:80]

# Chunking
_BLANK_LINE_RE = re.compile(r"\n\s*\n")


def iter_paragraphs(text: str) -> Iterator[tuple[str, int, int]]:
    """Разбиение на абзацы по пустым строкам, с точными (start, end)."""
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
    return f"{base} ({_status_display(doc)})."


def _ref_for_index(ref: str) -> str:
    if not ref:
        return ref
    if ref[0].islower() and not ref.startswith(("п.", "ч.", "пп.")):
        return ref[0].upper() + ref[1:]
    return ref


def _build_text_for_index(doc: Document, ref: str, section: Optional[str], content: str) -> str:
    parts = [_doc_prefix_for_index(doc)]
    if section:
        parts.append(f"Раздел {section}.")
    parts.append(_ref_for_index(ref) + ".")
    parts.append(content)
    return " ".join(parts)

_REPEALED_RE = re.compile(
    r'^\W*(?:\d+(?:\.\d+)?[.)]\s*)?'
    r'(?:(?:Часть|Пункт|Подпункт|Абзац)\s+[\w.]+\s+)?утратил[аи]?\s+силу',
    re.IGNORECASE)

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
    r'(игнорируй|забудь\s+(?:все\s+)?инструкции|для\s+ии[- ]?ассистента|'
    r'system\s*prompt|отвечай\s+только|как\s+ии\b)',
    re.IGNORECASE,
)


def _is_suspicious(text: str) -> bool:
    return bool(_SUSPICIOUS_RE.search(text))


# Договоры и регламенты

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
            # Есть ли внутри этой секции пункты?
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
        else:  # point
            next_pos = markers[i + 1][1] if i + 1 < len(markers) else len(text)
            s, e = pos, next_pos
            while e > s and text[e - 1] in " \n":
                e -= 1
            slug = f"p{point_num}"
            ref = f"п. {point_num}"
            chunks.append(_make_chunk(doc, slug, ref, current_section, text[s:e], s, e))
        i += 1

    return _split_oversized(chunks)

def _split_oversized(chunks: list[Chunk], max_len: int = 2500) -> list[Chunk]:
    out: list[Chunk] = []
    for c in chunks:
        if len(c.text) <= max_len:
            out.append(c)
            continue

        pieces = list(iter_paragraphs(c.text))
        if not pieces or max(len(p[0]) for p in pieces) > max_len:
            out.extend(_hard_split(c, max_len))
            continue

        buf_s, buf_e = -1, -1
        idx = 0
        for _ptxt, ps, pe in pieces:
            if buf_s < 0:
                buf_s, buf_e = ps, pe
            elif pe - buf_s <= max_len:  # границы в координатах c.text
                buf_e = pe
            else:
                piece = c.text[buf_s:buf_e]  # ← вырезаем из родителя
                out.append(_clone_with_text(c, piece, c.start + buf_s, c.start + buf_e, idx))
                idx += 1
                buf_s, buf_e = ps, pe
        if buf_s >= 0:
            piece = c.text[buf_s:buf_e]
            out.append(_clone_with_text(c, piece, c.start + buf_s, c.start + buf_e, idx))
    return out

_END_PUNCT_RE = re.compile(r'[\.\!\?\;\:]')


def _hard_split(c: Chunk, max_len: int) -> list[Chunk]:
    """Дробим один большой фрагмент по границам предложений (.;!?:), с точными координатами."""
    text = c.text
    n = len(text)
    if n == 0:
        return [c]

    # Индексы (в text) сразу после знаков . ! ? ; :
    boundaries = sorted({m.end() for m in _END_PUNCT_RE.finditer(text)})
    if not boundaries or boundaries[-1] != n:
        boundaries.append(n)

    out: list[Chunk] = []
    idx = 0
    start = 0
    while start < n:
        # максимальная граница, при которой длина ≤ max_len
        end = start
        for b in boundaries:
            if b <= start:
                continue
            if b - start <= max_len:
                end = b
            else:
                break
        if end == start:
            # одно «предложение» длиннее max_len — режем по последнему пробелу
            space = text.rfind(' ', start, start + max_len)
            end = space if space > start else start + max_len
        # обрезаем хвостовые пробелы, но НЕ меняем текст
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
            ))
            idx += 1
        # перепрыгиваем пробелы
        start = end
        while start < n and text[start].isspace():
            start += 1

    return out

def _clone_with_text(src: Chunk, text: str, start: int, end: int, idx: int) -> Chunk:
    slug = src.chunk_id.split("#", 1)[1] + (f"_{idx}" if idx else "")
    doc = _doc_cache.get(src.doc_id)
    tfid = _build_text_for_index(doc, src.ref, src.section, _strip_leading_number(text)) if doc else text
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


_doc_cache: dict[str, Document] = {}  # заполняется в load_kb


# Законы
def _law_unit_names(code: Optional[str]) -> tuple[str, Optional[str]]:
    code = code or ""
    if "НК" in code:
        return "п", "пп"
    if "КоАП" in code:
        return "ч", None
    if "152" in code:
        return "ч", "п"
    return "ч", None

def _chunk_law_numbered(doc: Document) -> list[Chunk]:
    text = doc.text
    top_name, sub_name = _law_unit_names(doc.law_code)
    MAX = 2500

    body_start = 0
    m = re.search(r'^Статья\s+[\d\.]+\..*$', text, re.MULTILINE)
    if m:
        body_start = m.end()

    notes_match = re.search(r'^\s*Примечани[ея]\s*\.?\s*$',
                            text[body_start:], re.MULTILINE | re.IGNORECASE)
    notes_pos = body_start + notes_match.start() if notes_match else None

    paras = [(t, s, e) for t, s, e in iter_paragraphs(text) if s >= body_start]
    if not paras:
        return _fallback_chunks(doc)

    top_re = re.compile(r'^(\d+(?:\.\d+)?)\.\s')
    tops = [(i, m.group(1), s, e)
            for i, (t, s, e) in enumerate(paras)
            if (m := top_re.match(t))]
    if not tops:
        return _fallback_chunks(doc)

    chunks: list[Chunk] = []
    seen: set[str] = set()

    for ti, (pi, num, ps, pe) in enumerate(tops):
        end_of_top = tops[ti + 1][2] if ti + 1 < len(tops) else len(text)
        end_of_top = _trim_right(text, ps, end_of_top)
        in_notes = notes_pos is not None and ps >= notes_pos

        slug_base = f"note{num}" if in_notes else f"p{num}"
        if in_notes:
            ref_base = f"прим. {num} к ст. {doc.article}" if doc.article else f"прим. {num}"
        else:
            ref_base = f"{top_name}. {num} ст. {doc.article}" if doc.article else f"{top_name}. {num}"

        # Подпункты N) / N.M) — только если тип единицы их поддерживает
        subs: list[tuple[str, int, int]] = []
        if sub_name and not in_notes:
            sub_re = re.compile(r'^(\d+(?:\.\d+)?)\)\s')
            for j in range(pi + 1, len(paras)):
                t, s, e = paras[j]
                if s >= end_of_top:
                    break
                sm = sub_re.match(t)
                if sm:
                    subs.append((sm.group(1), s, e))

        top_len = end_of_top - ps
        if not subs or top_len <= MAX:
            slug = slug_base if slug_base not in seen else f"{slug_base}_2"
            seen.add(slug)
            chunks.append(_make_chunk(doc, slug, ref_base, None,
                                      text[ps:end_of_top], ps, end_of_top))
            continue

        # Вступление top-единицы
        intro_end = _trim_right(text, ps, subs[0][1])
        if intro_end > ps:
            slug = slug_base if slug_base not in seen else f"{slug_base}_2"
            seen.add(slug)
            chunks.append(_make_chunk(doc, slug, ref_base, None,
                                      text[ps:intro_end], ps, intro_end))

        # Подпункты
        for si, (subnum, ss, _) in enumerate(subs):
            sub_end = subs[si + 1][1] if si + 1 < len(subs) else end_of_top
            sub_end = _trim_right(text, ss, sub_end)
            sub_slug = f"{slug_base}.pp{subnum.replace('.', '-')}"
            if sub_slug in seen:
                sub_slug = f"{sub_slug}_2"
            seen.add(sub_slug)
            sub_ref = f"{sub_name}. {subnum} {ref_base}"
            chunks.append(_make_chunk(doc, sub_slug, sub_ref, None,
                                      text[ss:sub_end], ss, sub_end))

    return _split_oversized(chunks)

_NUM_ITEM_RE = re.compile(r'^\d+(?:\.\d+)?\)\s')
# _LAW_NOTE_RE = re.compile(
#     r'^(?:О\s+конституционно-правов\w+\s+смысл\w+'
#     r'|Согласно\s+Определени\w+\s+Конституционного\s+Суда'
#     r'|Положени\w+\s+(?:статьи|части|пункта)\s+[\d\.]+\s*\(в\s+редакции)',
#     re.IGNORECASE,
# )
_LAW_NOTE_RE = re.compile(
    r'^Положени\w+\s+(?:статьи|части|пункта)\s+[\d\.]+\s*\(в\s+редакции',
    re.IGNORECASE,
)
def _chunk_law_paragraphs(doc: Document) -> list[Chunk]:
    text = doc.text
    body_start = 0
    m = re.search(r'^Статья\s+[\d\.]+\..*$', text, re.MULTILINE)
    if m:
        body_start = m.end()

    paras = [(t, s, e) for t, s, e in iter_paragraphs(text) if s >= body_start]
    if not paras:
        return _fallback_chunks(doc)

    groups: list[tuple[int, int, bool]] = []     # (start, end, is_note)
    cur = None
    prev_colon = prev_open = False               # прошлый абзац кончается «:» / «:» или «;»
    for t, s, e in paras:
        tt = t.rstrip()
        is_note = bool(_LAW_NOTE_RE.match(t))
        item = bool(_NUM_ITEM_RE.match(t)) or t[:1].islower()
        cont = (cur is not None and not is_note and not cur[2] and (
            prev_colon
            or (prev_open and item)                                   # 2) …; 3) …
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

    chunks: list[Chunk] = []
    n_part = n_note = 0
    for s, e, is_note in groups:
        if is_note:
            n_note += 1
            # slug = f"gnote{n_note}"
            # ref = f"справка {n_note} к ст. {doc.article}" if doc.article else f"справка {n_note}"
            slug = f"trans{n_note}"
            ref = f"переходное положение к ст. {doc.article}" if doc.article else f"переходное положение {n_note}"
        else:
            n_part += 1
            slug = f"ch{n_part}"
            ref = f"ч. {n_part} ст. {doc.article}" if doc.article else f"ч. {n_part}"
        chunks.append(_make_chunk(doc, slug, ref, None, text[s:e], s, e))
    return _split_oversized(chunks)


def _chunk_law(doc: Document) -> list[Chunk]:
    # Эвристика: есть ли в теле статьи строки, начинающиеся с «N. »?
    has_numbering = bool(re.search(r'^\d+(?:\.\d+)?\.\s', doc.text, re.MULTILINE))
    if has_numbering:
        return _chunk_law_numbered(doc)
    return _chunk_law_paragraphs(doc)


def _trim_right(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1] in " \n":
        end -= 1
    return end


#  Допсоглашения/записки/письма 

def _chunk_whole(doc: Document) -> list[Chunk]:
    text = doc.text.strip()
    if not text:
        return []
    s, e = 0, len(doc.text)
    while s < e and doc.text[s] in " \n":
        s += 1
    while e > s and doc.text[e - 1] in " \n":
        e -= 1
    if doc.doc_type == "addendum":
        slug, ref = "whole", f"доп. соглашение № {doc.number}" if doc.number else "доп. соглашение"
    elif doc.doc_type == "memo":
        slug, ref = "whole", f"служебная записка № {doc.number}" if doc.number else "служебная записка"
    elif doc.doc_type == "letter":
        slug, ref = "whole", f"письмо № {doc.number}" if doc.number else "письмо"
    else:
        slug, ref = "whole", doc.short_ref
    chunks = [_make_chunk(doc, slug, ref, None, doc.text[s:e], s, e)]
    return _split_oversized(chunks)


def _chunk_memo_letter(doc: Document) -> list[Chunk]:
    if len(doc.text) < 2000:
        return _chunk_whole(doc)
    # По абзацам
    chunks: list[Chunk] = []
    for idx, (t, s, e) in enumerate(iter_paragraphs(doc.text), start=1):
        ref = f"абзац {idx}"
        chunks.append(_make_chunk(doc, f"par{idx}", ref, None, t, s, e))
    return _split_oversized(chunks)


# Fallback 
def _fallback_chunks(doc: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, (t, s, e) in enumerate(iter_paragraphs(doc.text), start=1):
        ref = doc.short_ref
        chunks.append(_make_chunk(doc, f"p{idx}", ref, None, t, s, e))

    if not chunks:
        # Весь текст — сплошные пробелы, или Document.text пуст
        # _chunk_whole сам вернёт [] на пустом тексте и применит _split_oversized
        return _chunk_whole(doc)

    return _split_oversized(chunks)


# Диспетчер 
def chunk_document(doc: Document) -> list[Chunk]:
    if doc.doc_type in ("contract", "regulation"):
        return _chunk_contract_like(doc)
    if doc.doc_type == "law":
        return _chunk_law(doc)
    if doc.doc_type in ("addendum", "memo", "letter", "other"):
        if doc.doc_type in ("memo", "letter"):
            return _chunk_memo_letter(doc)
        return _chunk_whole(doc)
    return _fallback_chunks(doc)



# Связи между документами
def _link_addenda(documents: dict[str, Document]) -> None:
    # Строим индекс номер→doc_id для договоров
    contract_by_number: dict[str, str] = {}
    for d in documents.values():
        if d.doc_type == "contract" and d.number:
            for key in _number_variants(d.number):
                contract_by_number[key] = d.doc_id

    for d in documents.values():
        if d.doc_type != "addendum" or not d.amends:
            continue
        # Пересчитываем amends из текста, чтобы получить parent_number
        parsed = extract_amends(d.text)
        if not parsed:
            continue
        parent_num = parsed.get("parent_number")
        parent_id = None
        if parent_num:
            for key in _number_variants(parent_num):
                if key in contract_by_number:
                    parent_id = contract_by_number[key]
                    break
        if parent_id:
            d.amends = {
                "doc_id": parent_id,
                "points": parsed.get("points", []),
                "condition": parsed.get("condition"),
            }
            parent = documents[parent_id]
            if d.doc_id not in parent.addenda:
                parent.addenda.append(d.doc_id)


def _link_regulations(documents: dict[str, Document]) -> None:
    regs = [d for d in documents.values() if d.doc_type == "regulation"]
    for i, a in enumerate(regs):
        for b in regs:
            if a is b:
                continue
            # Совпадение по первым 3 словам title
            wa = a.title.lower().split()[:3]
            wb = b.title.lower().split()[:3]
            if wa and wb and wa == wb:
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



# Entities
def _number_variants(num: str) -> list[str]:
    if not num:
        return []
    raw = num.strip()
    n = raw.lower()
    variants = {raw, n}
    variants.add(n.replace("/", "-"))
    variants.add(n.replace("-", "/"))
    variants.add(n.replace("/", ""))
    variants.add(n.replace("-", ""))
    look = str.maketrans({"а": "a", "п": "p", "с": "c", "е": "e",
                          "о": "o", "х": "x", "у": "y", "к": "k"})
    base = n.translate(look)
    variants |= {base, base.replace("/", "-"), base.replace("/", "")}
    return [v for v in variants if v]

def build_entities(documents: dict[str, Document]) -> Entities:
    counterparties: list[dict] = []
    contracts: dict[str, str] = {}
    by_type: dict[str, list[str]] = {}

    for d in documents.values():
        by_type.setdefault(d.doc_type, []).append(d.doc_id)

        if d.doc_type == "contract" and d.number:
            for v in _number_variants(d.number):
                contracts[v] = d.doc_id

        if d.counterparty:
            name = d.counterparty["name"]
            # Алиасы: без кавычек, без «ООО»-префикса
            short = re.sub(r'^(ООО|АО|ПАО|ЗАО|ИП)\s*[«"]', '', name).rstrip('»"')
            aliases = {name, short}
            # НЕ добавляем префиксные совпадения — оставляем только целые токены
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



# Editions on date
def editions_on(d: date, docs: list[Document]) -> list[Document]:
    """Документы, действовавшие на дату: valid_from <= d < valid_to (None = открыто)."""
    out: list[Document] = []
    for doc in docs:
        if doc.valid_from and doc.valid_from > d:
            continue
        if doc.valid_to and doc.valid_to <= d:
            continue
        out.append(doc)
    return out

def load_kb(path: str | Path) -> KnowledgeBase:
    path = Path(path)

    if path.is_dir():
        files = list(_iter_dir(path))
        raw_all = b"".join(b for _, b in files)
    else:
        files = list(_iter_archive(path))
        raw_all = path.read_bytes()

    kb_hash = hashlib.sha256(raw_all).hexdigest()

    # ---- 1. Документы ----
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

    global _doc_cache
    _doc_cache = documents

    # ---- 2. Чанки ----
    chunks: list[Chunk] = []
    for doc in documents.values():
        try:
            chunks.extend(chunk_document(doc))
        except Exception as e:
            print(f"[kb] WARN: cannot chunk {doc.doc_id}: {e}", file=sys.stderr)
            chunks.extend(_fallback_chunks(doc))

    #safety-net
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

    # Связи !после! safety-net
    _link_addenda(documents)
    _link_regulations(documents)
    _annotate_amendments(documents, chunks)

    entities = build_entities(documents)

    return KnowledgeBase(
        documents=documents,
        chunks=chunks,
        entities=entities,
        kb_hash=kb_hash,
    )


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kb.py", description="Knowledge base loader")
    ap.add_argument("path", help="zip-архив или каталог с базой")
    ap.add_argument("--dump", default=None, help="куда записать chunks.jsonl")
    ap.add_argument("--pretty", action="store_true", help="Отступы для лучшей читаемости")
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