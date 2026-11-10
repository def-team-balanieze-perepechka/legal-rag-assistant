"""Тесты модуля core/kb.py по критериям приёмки из §6 ТЗ.

Соответствие тестов и критериев:
    1. test_demo_ref_found, test_demo_q2_tk70_six_months, test_demo_q11_memo_whole
    2. test_chunk_text_matches_document_slice
    3. test_chunk_ids_unique, test_chunk_ids_stable
    4. test_law_no_garant_service_lines, test_law_keeps_norm_lines
    5. test_regulation_validity, test_editions_on
    6. test_addendum_amends
    7. test_alpha_trade_vs_alpha_trading, test_contracts_lookup
    8. test_no_empty_or_oversized_chunks
    9. test_robustness_renamed_and_docx
   10. test_cli_dump
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from kb import editions_on, load_kb  # noqa: E402

KB_PATH = ROOT / "data" / "knowledge-base.zip"


@pytest.fixture(scope="module")
def kb():
    return load_kb(KB_PATH)



# Критерий 1. Все демо-ссылки находятся
# (вопрос, doc_id, ref, подстрока, которая должна быть в тексте чанка)
DEMO_REFS = [
    (1,  "contract_a-08-2024",           "п. 3.1", None),
    (1,  "contract_a-08-2024",           "п. 3.2", None),
    (1,  "contract_a-08-2024",           "п. 3.3", None),
    (4,  "contract_a-08-2024",           "п. 3.1", None),
    (4,  "contract_a-08-2024",           "п. 3.4", None),
    (5,  "reglament_soglasovaniya_2023", "п. 2.4", None),
    (6,  "contract_p-17-2024",           "п. 6.1", None),
    (6,  "contract_p-17-2024",           "п. 3.3", None),
    (7,  "contract_a-08-2024",           "п. 4.3", None),
    (8,  "contract_p-17-2024",           "п. 6.2", None),
    (9,  "contract_pt-05-2023",          "п. 4.1", None),
    (9,  "contract_pt-05-2023",          "п. 4.2", None),
    (10, "reglament_soglasovaniya_2025", "п. 1.3", None),
    (12, "tk_70",                        "ч. 3 ст. 70", "В период испытания"),
    (13, "152fz_9",                      "ч. 2 ст. 9",  "Согласие на обработку"),
    (14, "tk_59",                        "ч. 1 ст. 59", "Срочный трудовой договор заключается"),
    (15, "contract_a-08-2024",           "п. 6.3", None),
    (16, "contract_p-17-2024",           "п. 2.1", None),
    (17, "contract_pt-05-2023",          "п. 6.1", None),
    (18, "reglament_soglasovaniya_2025", "п. 3.2", None),
]


@pytest.mark.parametrize(
    "q,doc_id,ref,needle",
    DEMO_REFS,
    ids=[f"q{q}-{d}-{r}" for q, d, r, _ in DEMO_REFS],
)
def test_demo_ref_found(kb, q, doc_id, ref, needle):
    assert doc_id in kb.documents, f"нет документа {doc_id}"
    c = kb.find_ref(doc_id, ref)
    assert c is not None, f"find_ref({doc_id!r}, {ref!r}) вернул None"
    assert c.text.strip(), "пустой чанк"
    if needle:
        assert needle in c.text, f"в чанке {c.chunk_id} нет «{needle}»"
    elif ref.startswith("п. "):
        # чанк пункта договора/регламента начинается со своего номера
        assert c.text.lstrip().startswith(ref[3:]), (c.chunk_id, c.text[:40])


def test_demo_q2_tk70_six_months(kb):
    """Вопрос 2: отдельный чанк ТК ст. 70 со сроком 6 месяцев находится по ref."""
    hits = [c for c in kb.chunks if c.doc_id == "tk_70" and "шести месяцев" in c.text]
    assert hits, "ни один чанк tk_70 не содержит «шести месяцев»"
    assert kb.find_ref("tk_70", hits[0].ref) is not None


def test_demo_q11_memo_whole(kb):
    """Вопрос 11: служебная записка ЮО-7 — весь документ одним чанком."""
    doc = kb.documents["memo_yuo-7"]
    cs = [c for c in kb.chunks if c.doc_id == "memo_yuo-7"]
    assert len(cs) == 1
    assert cs[0].text == doc.text[cs[0].start:cs[0].end]
    assert len(cs[0].text) >= 0.9 * len(doc.text)



# Критерий 2. text == doc.text[start:end] для каждого чанка

def test_chunk_text_matches_document_slice(kb):
    for c in kb.chunks:
        doc = kb.documents[c.doc_id]
        actual = doc.text[c.start:c.end]
        assert actual == c.text, (
            f"slice mismatch in {c.chunk_id}: "
            f"doc.text[{c.start}:{c.end}] (len {len(actual)}) "
            f"!= chunk.text (len {len(c.text)})"
        )



# Критерий 3. chunk_id уникальны и стабильны между запусками

def test_chunk_ids_unique(kb):
    ids = [c.chunk_id for c in kb.chunks]
    assert len(ids) == len(set(ids)), (
        "дубли chunk_id: "
        + str([x for x in set(ids) if ids.count(x) > 1][:5])
    )


def test_chunk_ids_stable():
    a = load_kb(KB_PATH)
    b = load_kb(KB_PATH)
    assert [c.chunk_id for c in a.chunks] == [c.chunk_id for c in b.chunks]



# Критерий 4. Законы очищены от служебных строк Гаранта, нормы сохранены

FORBIDDEN_LAW_LINE = re.compile(
    r"^\s*(?:Статья|Пункт|Подпункт|Часть|Абзац)\s+[\d\.]+\s+"
    r"(?:изменен[аоы]?|дополнен[аоы]?|утратил[аи]?\s+силу)"
    r"|^\s*Федеральн(?:ым|ыми)\s+законом\s+от"
    r"|^\s*См\.\s+текст"
    r"|^\s*Информация\s+об\s+изменениях"
    r"|^\s*О\s+проверке\s+конституционности",
    re.IGNORECASE | re.MULTILINE,
)


def test_law_no_garant_service_lines(kb):
    for c in kb.chunks:
        if kb.documents[c.doc_id].doc_type != "law":
            continue
        m = FORBIDDEN_LAW_LINE.search(c.text)
        assert m is None, (
            f"служебная строка Гаранта в {c.chunk_id}: {m.group(0)!r}"
        )

def test_law_keeps_norm_lines(kb):
    """«Положения настоящего пункта/подпункта/абзаца …» — норма, её нельзя вырезать."""
    joined = "\n".join(
        c.text for c in kb.chunks
        if kb.documents[c.doc_id].doc_type == "law"
    )
    assert "Положения настоящего" in joined, (
        "Все «Положения настоящего …» были вырезаны при чистке"
    )



# Критерий 5. Регламенты: статусы, даты, editions_on

def test_regulation_validity(kb):
    r23 = kb.documents["reglament_soglasovaniya_2023"]
    r25 = kb.documents["reglament_soglasovaniya_2025"]

    assert r23.status == "expired"
    assert r23.valid_to == date(2025, 3, 1)

    assert r25.status == "active"
    assert r25.valid_from == date(2025, 3, 1)


def test_editions_on(kb):
    docs = list(kb.documents.values())

    ids_2024 = {d.doc_id for d in editions_on(date(2024, 8, 15), docs)}
    assert "reglament_soglasovaniya_2023" in ids_2024
    assert "reglament_soglasovaniya_2025" not in ids_2024

    ids_2025 = {d.doc_id for d in editions_on(date(2025, 6, 1), docs)}
    assert "reglament_soglasovaniya_2025" in ids_2025
    assert "reglament_soglasovaniya_2023" not in ids_2025



# Критерий 6. Допсоглашение: amends и amended_by

def test_addendum_amends(kb):
    adds = [d for d in kb.documents.values() if d.doc_type == "addendum"]
    assert adds, "в базе нет ни одного допсоглашения"
    add = adds[0]

    assert add.amends is not None, "amends не заполнен"
    assert add.amends.get("doc_id") == "contract_p-17-2024"
    assert add.amends.get("points") == ["3.2"]

    cond = add.amends.get("condition") or ""
    assert "1 октября 2024" in cond, (
        f"условие применения не извлечено: {cond!r}"
    )

    parent = kb.documents["contract_p-17-2024"]
    assert add.doc_id in parent.addenda

    chunk = kb.find_ref("contract_p-17-2024", "п. 3.2")
    assert chunk is not None
    assert add.doc_id in chunk.meta.get("amended_by", [])



# Критерий 7. Сущности: Альфа-Трейд ≠ Альфа-Трейдинг, contracts lookup

def _cp_by_name(kb, name_part: str):
    return next(
        (c for c in kb.entities.counterparties if name_part in c["name"]),
        None,
    )


def test_alpha_trade_vs_alpha_trading(kb):
    trade = _cp_by_name(kb, "Альфа-Трейд")
    trading = _cp_by_name(kb, "Альфа-Трейдинг")

    assert trade is not None, "ООО «Альфа-Трейд» не найдено в сущностях"
    assert trading is not None, "ООО «Альфа-Трейдинг» не найдено в сущностях"

    assert trade["name"] != trading["name"]
    assert set(trade["doc_ids"]).isdisjoint(trading["doc_ids"])
    assert set(trade["aliases"]).isdisjoint(trading["aliases"]), (
        f"алиасы пересекаются: {set(trade['aliases']) & set(trading['aliases'])}"
    )


def test_contracts_lookup(kb):
    c = kb.entities.contracts
    assert c.get("П-17/2024") == "contract_p-17-2024"
    assert c.get("п-17-2024") == "contract_p-17-2024"
    assert c.get("ПТ-05/2023") == "contract_pt-05-2023"
    assert c.get("П-17/2024") != c.get("ПТ-05/2023")



# Критерий 8. Размеры чанков

def test_no_empty_or_oversized_chunks(kb):
    for c in kb.chunks:
        assert c.text.strip(), f"пустой чанк {c.chunk_id}"
        assert len(c.text) <= 2500, (
            f"{c.chunk_id}: {len(c.text)} символов (> 2500)"
        )



# Критерий 9. Устойчивость: переименование без префикса группы + DOCX

@pytest.fixture(scope="module")
def modified_kb(tmp_path_factory):
    """Копия базы, где один договор переименован без префикса группы,
    плюс один DOCX-договор (если доступен python-docx).

    Возвращает (kb, has_docx).
    """
    src = KB_PATH
    dst = tmp_path_factory.mktemp("kbmod") / "kb.zip"
    has_docx = False

    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for name in zin.namelist():
            data = zin.read(name)
            # Переименовываем «документы компании__Договор аренды ...» → без префикса
            if "Договор аренды" in name and "__" in name:
                zout.writestr(name.split("__", 1)[1], data)
            else:
                zout.writestr(name, data)

        # Опционально — DOCX-договор
        try:
            import docx  # type: ignore

            d = docx.Document()
            for p in [
                "ДОГОВОР ОКАЗАНИЯ УСЛУГ № Т-99/2024",
                "г. Москва — 10 февраля 2024 г.",
                "ООО «Тест-Сервис» (ИНН 1234567890), «Исполнитель», "
                "и ООО «Меридиан» (ИНН 7709876543), «Заказчик»,",
                "заключили договор.",
                "1. Предмет договора",
                "1.1. Исполнитель оказывает услуги.",
            ]:
                d.add_paragraph(p)
            buf = io.BytesIO()
            d.save(buf)
            zout.writestr("extra_contract.docx", buf.getvalue())
            has_docx = True
        except ImportError:
            pass

    return load_kb(dst), has_docx


def test_robustness_renamed_and_docx(modified_kb):
    kb, has_docx = modified_kb

    # Переименованный договор аренды всё ещё распознан как contract
    # (тип определён по тексту, а не по префиксу группы)
    renamed = [
        d for d in kb.documents.values()
        if "аренды" in d.title.lower() and d.doc_type == "contract"
    ]
    assert renamed, "договор аренды не распознан после переименования"
    assert renamed[0].number == "А-08/2024"

    # DOCX-договор: распознан как contract, извлечены номер и контрагент
    if has_docx:
        docx_docs = [
            d for d in kb.documents.values()
            if d.filename.endswith(".docx")
        ]
        assert docx_docs, "DOCX-договор не попал в kb"
        d = docx_docs[0]
        assert d.doc_type == "contract", (
            f"DOCX распознан как {d.doc_type!r}, ожидался 'contract'"
        )
        assert d.number == "Т-99/2024"
        assert d.counterparty is not None
        assert "Тест-Сервис" in d.counterparty["name"]
        # У него тоже есть чанки
        assert any(c.doc_id == d.doc_id for c in kb.chunks)

    # Общие инварианты для всех чанков модифицированной базы
    for c in kb.chunks:
        doc = kb.documents[c.doc_id]
        assert doc.text[c.start:c.end] == c.text
        assert c.text.strip()
        assert len(c.text) <= 2500



# Критерий 10. CLI: python core/kb.py <zip> --dump chunks.jsonl


def test_cli_dump(tmp_path):
    out = tmp_path / "chunks.jsonl"

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "core" / "kb.py"),
            str(KB_PATH),
            "--dump",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"CLI упал:\nstdout={r.stdout}\nstderr={r.stderr}"
    assert out.exists(), "chunks.jsonl не создан"

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines, "chunks.jsonl пуст"

    # Каждая строка — валидный JSON с обязательными ключами
    required = {"chunk_id", "doc_id", "ref", "text", "start", "end"}
    for ln in lines:
        obj = json.loads(ln)
        assert required <= obj.keys(), (
            f"в строке нет полей {required - obj.keys()}: {obj.get('chunk_id')}"
        )

    # Инвариант start:end сохраняется и в выгрузке
    kb = load_kb(KB_PATH)
    for ln in lines:
        c = json.loads(ln)
        doc = kb.documents[c["doc_id"]]
        assert doc.text[c["start"]:c["end"]] == c["text"], (
            f"в дампе нарушен инвариант слайса: {c['chunk_id']}"
        )


def test_split_oversized_preserves_slice():
    """Синтетика: чанк > 2500 с 'пустыми' переводами строк между абзацами —
    после нарезки text == doc.text[start:end] на каждом куске."""
    from kb import Document, Chunk, _split_oversized
    body = ("A" * 1200 + "\n\n\n" + "B" * 1200 + "\n  \n" + "C" * 1200)
    doc = Document(doc_id="d", doc_type="other", title="t",
                   short_ref="t", text=body)
    src = Chunk(chunk_id="d#x", doc_id="d", ref="x", section=None,
                text=body, text_for_index="x", start=0, end=len(body))
    pieces = _split_oversized([src])
    assert len(pieces) >= 2, "фрагмент 3600 симв. должен быть разрезан"
    for p in pieces:
        assert body[p.start:p.end] == p.text, (
            f"нарушен слайс: {p.chunk_id} len {len(p.text)}"
        )
        assert len(p.text) <= 2500


def test_152fz_part4_has_subchunks(kb):
    """ТЗ §3.4: для 152-ФЗ подпункты ч. 4 ст. 9 — отдельные чанки."""
    c = kb.find_ref("152fz_9", "п. 1 ч. 4 ст. 9")
    assert c is not None, "нет чанка п. 1 ч. 4 ст. 9"
    assert "фамилию, имя, отчество" in c.text

    # родительский чанк ч. 4 ст. 9 тоже остался (intro)
    parent = kb.find_ref("152fz_9", "ч. 4 ст. 9")
    assert parent is not None


def test_law_meta_edition_is_actual(kb):
    """ТЗ §3.3: для законов edition == 'действующая'."""
    for d in kb.documents.values():
        if d.doc_type == "law":
            assert d.edition == "действующая", (
                f"{d.doc_id}: edition={d.edition!r}, ожидалось 'действующая'"
            )