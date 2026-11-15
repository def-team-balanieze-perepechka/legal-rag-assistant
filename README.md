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

Если эмбеддер или реранкер недоступны, поиск продолжается на BM25 и RRF. Если модель дважды не выдала JSON (обрыв по лимиту, сломанный формат), делается запасной вызов: короткий промпт без JSON по 3 лучшим фрагментам (`LLM_NO_THINK=1` — без рассуждений для qwen3). Если не отвечает LLM, ассистент честно сообщает, что ответ сформировать не удалось; ошибка одного вопроса не роняет прогон.

Проверки после генерации (кодом): ссылки на пункты, которых нет среди переданных модели фрагментов, удаляются; к отказам («нет в базе», вне темы, противоправный запрос) подпись «Источник:» не добавляется. У статей закона есть тема-налог (НДС / НДФЛ / налог на прибыль), определенная по тексту: она входит в индекс и в заголовок фрагмента, а если в вопросе назван налог, статьи о других налогах из выдачи убираются.

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

## Локальный судья

Сравнивает версию агента с baseline (стартовый `solution.py` из `data/sample_submission.zip`) по методике платформы: попарно с опорой на эталон, два раунда со сменой порядка, `Score = 0.7·MatchRate + 0.3·WinRate`, выдуманный факт — 0 за вопрос; заглушка агента — 0 без вызова судьи. Кейсы: демо из `data/` и свои из `eval/cases_own.csv`.

```bash
python eval/judge.py --name v1 --checks-only   # только ответы агента и проверки кодом (заглушки, несуществующие пункты, ложные отказы)
python eval/judge.py --name v1                 # полный прогон: агент, baseline, судья, отчет
```

- Отчет: `out/eval/<name>/report.md`, история прогонов: `out/eval/history.csv` (разница с прошлым прогоном на том же наборе кейсов).
- Ответы и вердикты кешируются: повторный запуск с тем же `--name` не вызывает LLM; для новой версии агента — новое `--name`.
- Фильтры: `--ids demo-5 own-3`, `--scenario false_premise`, `--source demo|own`.
- Судья: `JUDGE_MODEL` (по умолчанию `qwen3:8b`), `JUDGE_BASE_URL` / `JUDGE_API_KEY` (по умолчанию как у агента), `JUDGE_MAX_TOKENS` (8192: qwen3:8b с 4096 иногда не успевает дорассуждать), `JUDGE_TIMEOUT` (600 с). На локальной Ollama генератор и эмбеддер выгружаются перед фазой судьи. Baseline запускается пачками по 10 вопросов без настроек агента из `.env` (`TOP_K` и т. п.), с лимитом `BASELINE_MAX_TOKENS` (по умолчанию 4096: с 1024 локальная qwen3 не успевает ответить).
- Ограничение: наш судья не совпадает с судьей платформы.



## Запуск через Docker

Проект поддерживает два режима работы в Docker:

1. **Веб-интерфейс (UI + API)** — интерактивный ассистент для работы с базой знаний в браузере.
2. **Пакетный прогон (Solution)** — фоновая обработка файла `questions.csv` для генерации `submission.csv` (используется для сдачи на платформе или массового тестирования).

Для изоляции зависимостей проект использует три разных файла `requirements` (`requirements.txt` для solution, `requirements-api.txt` для API, `requirements-ui.txt` для UI).

### Требования

- **Docker** с **Compose ≥ 2.24** (нужен для `env_file: required: false`). Проверка: `docker compose version`.
- **Ollama** или **любой OpenAI-совместимый API**  на хосте (см. раздел «Внешние API»).
- Для локальной Ollama — скачанные модели (см. раздел «Ollama на хосте»).

### Подготовка окружения

```bash
# 1) Скопируйте шаблон окружения
cp .env.example .env

# 2) Создайте каталоги для входа и результата
mkdir -p data out
```

На Linux/WSL — чтобы контейнер писал в `./out` от вашего пользователя, а не от `root`:

```bash
export LOCAL_UID=\$(id -u)
export LOCAL_GID=\$(id -g)
```

Положите входные данные в `data/`:
- `data/questions.csv` (для пакетного режима)
- `data/knowledge-base.zip` (база знаний)

### Режим 1: Запуск всего решения (Веб-интерфейс + API)

В этом режиме поднимаются контейнеры `ui` (Streamlit) и `api` (FastAPI). 
```bash
# Сборка и запуск UI + API (по умолчанию)
docker compose up --build
```

После запуска:
- **Веб-интерфейс:** доступен по адресу `http://localhost:8000`.
- **API бэкенд:** доступен только с вашей машины по адресу `http://127.0.0.1:8001`.

API начнет отвечать только после того, как загрузится база знаний (Docker healthcheck будет ждать ответа от `/health`). Чтобы остановить всё, нажмите `Ctrl+C` или выполните `docker compose down`.

### Режим 2: Пакетный прогон (только Solution)

Если вам нужно просто обработать файл `questions.csv` и получить `submission.csv`, запускать UI и API не обязательно. Сервис `solution` находится в профиле `batch` и запускается по требованию.

```bash
# Сборка и запуск только solution
docker compose --profile batch run --rm solution
```

Флаг `--rm` автоматически удалит контейнер после завершения скрипта. Результат появится в файле `out/submission.csv`. В логах вы увидите процесс загрузки базы знаний, инициализации LLM и прогресс обработки вопросов (каждые 10 штук).

### Переменные окружения

Шаблон — `.env.example`. Скопируйте в `.env` и при необходимости поправьте.

| Переменная | Назначение | По умолчанию (в `.env.example`) |
| :--- | :--- | :--- |
| `LLM_BASE_URL` | Адрес OpenAI-совместимого API | `http://host.docker.internal:11434/v1` |
| `LLM_API_KEY` | Ключ (для локальной Ollama — любое непустое значение) | `ollama` |
| `LLM_MODEL` | Имя модели LLM | `qwen3:4b` |
| `EMB_MODEL` | Модель эмбеддингов | `bge-m3` |
| `RERANKER_MODEL` | Модель реранкера (пусто — отключен) | |
| `TOP_K` | Сколько чанков брать в контекст | `8` |
| `TEMPERATURE` | Температура генерации | `0` |
| `MAX_TOKENS` | Лимит токенов ответа | `2048` |
| `MAX_WORKERS` | Параллельные запросы (для solution) | `1` |
| `TIME_BUDGET_SEC` | Бюджет прогона solution, в секундах | `900` |
| `CHAT_TIMEOUT` | Таймаут одного запроса в UI/API, в секундах | `180` |
| `API_URL` | Адрес API для UI (задается в compose) | `http://api:8000` |

Дополнительно поддерживаются `LLM_TIMEOUT`, `LLM_EXTRA_BODY`, `DEV_EMBED_BASE_URL`, `DEV_RERANK_BASE_URL`, `EMBEDDER_API_KEY`, `RERANKER_API_KEY`.

### Внешние API

На платформе используются:

| Компонент | Сервис / модель | Переменная |
| :--- | :--- | :--- |
| LLM | DeepSeek V4.1 Flash | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` |
| Эмбеддинги | BAAI/bge-m3 | `EMB_MODEL` |
| Реранкер (необязательно) | Qwen3 Reranker 8B | `RERANKER_MODEL` |

Ключей в репозитории нет. Для подключения внешнего API раскомментируйте соответствующий блок в `.env.example` и заполните переменные.

### Порты

- **8000** — веб-интерфейс (Streamlit).
- **8001** — API (FastAPI). Доступен только на `127.0.0.1` (`localhost`), чтобы не светить ключи LLM в сеть.

### Зависимости

В проекте три файла зависимостей, чтобы Docker собирал легковесные образы:
- `requirements.txt` — для solution (`numpy`, `rank-bm25`, `httpx`, `python-docx`).
- `requirements-api.txt` — для бэкенда (`fastapi`, `uvicorn`, `pydantic`, `httpx`).
- `requirements-ui.txt` — для фронтенда (`streamlit`, `httpx`).

`Dockerfile` использует `ARG REQS` для установки нужного набора пакетов в каждый сервис.

### Проверка работоспособности

**Пакетный прогон:**

```bash
docker compose --profile batch run --rm solution
head -3 out/submission.csv
```

- **Ожидаемо:** файл содержит заголовок `id,answer` и ответы. В логах есть `dense=True` (если эмбеддер доступен) и нет `Connection refused`.

**Веб-интерфейс:**

```bash
docker compose up --build
```

- **Ожидаемо:**
  - В логах API: `Application startup complete` и `Uvicorn running on http://0.0.0.0:8000`.
  - В логах UI: `You can now view your Streamlit app`.
  - В браузере на `http://localhost:8000` открывается чат, и ассистент отвечает на вопросы.

### Ollama на хосте

По умолчанию `.env.example` настроен на Ollama на хосте. Ollama доступна по имени `host.docker.internal` (в `compose.yaml` уже прописан `extra_hosts: ["host.docker.internal:host-gateway"]`).

Скачайте модели:

```bash
ollama pull qwen3:4b   # LLM_MODEL
ollama pull bge-m3     # EMB_MODEL
```

Ollama должна слушать не только `127.0.0.1`, иначе из контейнера до неё не достучаться:
- **Windows:** `setx OLLAMA_HOST "0.0.0.0:11434"` и `setx OLLAMA_CONTEXT_LENGTH 8192`, затем перезапустите Ollama.
- **Linux (systemd):** отредактируйте `systemctl edit ollama`, добавив `Environment="OLLAMA_HOST=0.0.0.0:11434"` и `Environment="OLLAMA_CONTEXT_LENGTH=8192"`.

### Известные ограничения

- **Индекс строится заново при каждом запуске.** `solution.py` и API делают вызовы `/embeddings` к `EMB_MODEL`, это занимает время. Между запусками индекс не кешируется.
- **Без эмбеддера поиск работает только на BM25.** Если `EMB_MODEL` недоступен, система деградирует до BM25 (`dense=False` в логе), но продолжает работать.
- **Без реранкера поиск работает без него.** Для локальной Ollama `RERANKER_MODEL` оставляйте пустым.
- **Ответы на локальной модели хуже, чем на платформенной.** `qwen3:4b` — компромисс по размеру; на платформе используется DeepSeek V4.1 Flash.
- **Права на `out/` (Linux/WSL).** Контейнер работает от пользователя `LOCAL_UID:LOCAL_GID`. Если не делали `export LOCAL_UID=$(id -u)`, прогон может упасть с `PermissionError`.



# Legal RAG Assistant — API

HTTP-бекенд юридического ассистента: FastAPI поверх `core/` (RAG по нормативной базе знаний).

## Запуск


```bash
pip install -r requirements-api.txt
cp .env.example .env          # заполнить LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, DATA_DIR
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Сервис: http://localhost:8000 
Swagger: http://localhost:8000/docs

При старте загружается база знаний из DATA_DIR/knowledge-base.zip (или KB_PATH).
Пока идёт загрузка, /health отдаёт 503, после — 200.

