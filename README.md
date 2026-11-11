# legal-rag-assistant

Юридический ассистент по базе знаний компании (Hack&Change 2026, кейс ПСБ). Отвечает строго по документам базы и ссылается на конкретные пункты.

## Агент v1: гибридный RAG со строгими цитатами

```
вопрос ──► разбор диалога ──► поиск ──────────────────────────────► генерация ──► проверка цитат ──► ответ
                              BM25 (свой стеммер)  ┐                один вызов LLM   chunk_id только из
                              bge-m3 /embeddings   ├─ RRF ─► /rerank   JSON с        найденных фрагментов;
                              сущности: № договора,┘                   chunk_id      ссылку «Источник: …»
                              контрагент → документ целиком;                         собирает код
                              другие редакции того же пункта; допсоглашения
```

| Модуль (`core/`) | Что делает                                                                                                                         |
|------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `kb.py`          | разбор базы (TXT/MD/DOCX): чистка Гаранта, чанки по пунктам и частям, редакции, допсоглашения, справочник договоров и контрагентов |
| `stem.py`        | токенизация и стеммер русского для BM25                                                                                            |
| `search.py`      | BM25 + эмбеддинги -> RRF -> реранкер; распознавание договоров и контрагентов                                                       |
| `llm.py`         | клиент прокси: `/chat/completions`, `/embeddings`, `/rerank`, повторы, отключение недоступных сервисов                             |
| `prompts.py`     | системный промпт и шаблоны                                                                                                         |
| `generate.py`    | вызов LLM, разбор JSON, отбрасывание неизвестных `chunk_id`, ссылка на источник                                                    |
| `assistant.py`   | `Assistant.answer(question, history=None, allow_clarify=False) -> dict`                                                            |

Если эмбеддер или реранкер недоступны, поиск продолжается на BM25 и RRF. Если не отвечает LLM, ассистент честно сообщает, что ответ сформировать не удалось; ошибка одного вопроса не роняет прогон.

## Запуск

```bash
pip install -r requirements.txt pytest python-dotenv
cp .env.example .env        # Заполнить LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
python solution.py          # DATA_DIR/questions.csv + база -> submission.csv
python eval/run_demo.py     # Демо-вопросы -> out/demo_report.md (ответ рядом с эталоном)
python -m pytest eval       # тесты без сети
python scripts/build_submission.py   # dist/submission.zip, все файлы в корне архива
```

Локально через Ollama: `ollama pull qwen3:4b` и `ollama pull bge-m3`, настройки — в конце `.env.example`. Реранкера в Ollama нет: без `DEV_RERANK_BASE_URL` поиск работает без него.

## Запуск через Docker

Проект можно запустить одной командой в контейнере — так же, как он работает на платформе:
`DATA_DIR/questions.csv` + `DATA_DIR/knowledge-base.zip` → `SUBMISSION_PATH` (`out/submission.csv`).
Docker-окружение не меняет код (`solution.py`, `core/`) и не содержит секретов: ключи и настройки
передаются только через `.env`.

### Что нужно заранее

- **Docker** (Compose v2): `docker compose version` должен показывать 2.x.
- **Ollama** на хосте (или любой другой OpenAI-совместимый API — см. ниже).
- Для локальной Ollama — скачанные модели:
  ```bash
  ollama pull qwen3:4b     # LLM_MODEL
  ollama pull bge-m3       # EMB_MODEL (эмбеддер для индекса)