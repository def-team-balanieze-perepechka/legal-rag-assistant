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

Проект запускается в контейнере **одной командой** — так же, как на платформе:
`DATA_DIR/questions.csv` + `DATA_DIR/knowledge-base.zip` → `SUBMISSION_PATH` (`out/submission.csv`).
Docker-окружение не меняет код (`solution.py`, `core/`); ключи и настройки передаются только через `.env`,
в образ они не попадают.

### Требования

- **Docker** с **Compose ≥ 2.24** (нужен для `env_file: required: false`). Проверка: `docker compose version` — должно быть 2.24+.
- **Ollama** на хосте (или любой OpenAI-совместимый API — см. раздел «Внешние API»).
- Для локальной Ollama — скачанные модели (см. раздел «Ollama на хосте»).

### Быстрый старт

```bash
# 1) Окружение для Docker (внутри контейнера localhost — это сам контейнер,
#    поэтому Ollama на хосте доступна по host.docker.internal)
cp .env.docker.example .env

# 2) Каталоги для входа и результата
mkdir -p data out
```

На Linux/WSL — чтобы контейнер писал в `./out` от вашего пользователя, а не от root:

```bash
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
```

Положите входные данные в `data/`:

- `data/questions.csv`
- `data/knowledge-base.zip`

Для проверки подойдёт копия `demo_questions.csv` под именем `questions.csv`:

```bash
cp data/demo_questions.csv data/questions.csv      # Linux/macOS/WSL
copy data\demo_questions.csv data\questions.csv    # Windows (cmd/PowerShell)
```

Соберите образ и запустите пакетный прогон:

```bash
docker compose build
docker compose run --rm solution
```

Результат — `out/submission.csv` с заголовком `id,answer`.

### Переменные окружения

Шаблон — `.env.docker.example`. Скопируйте в `.env` и при необходимости поправьте. 

| Переменная | Назначение | По умолчанию (в `.env.docker.example`) | Обязательна |
|---|---|---|---|
| `LLM_BASE_URL` | адрес OpenAI-совместимого API | `http://host.docker.internal:11434/v1` | да |
| `LLM_API_KEY` | ключ (для локальной Ollama — любое непустое значение) | `ollama` | да |
| `LLM_MODEL` | имя модели LLM | `qwen3:4b` | да |
| `EMB_MODEL` | модель эмбеддингов | `bge-m3` | да |
| `RERANKER_MODEL` | модель реранкера | пусто - реранк отключён | нет |
| `TOP_K` | сколько чанков брать в контекст | `8` | нет |
| `TEMPERATURE` | температура генерации | `0` | нет |
| `MAX_TOKENS` | лимит токенов ответа | `2048` | нет |
| `MAX_WORKERS` | параллельные запросы к LLM | `1` | нет |
| `TIME_BUDGET_SEC` | бюджет всего прогона, с | `900` | нет |
| `DATA_DIR` | каталог с входными данными | `/data` (переопределяется в `compose.yaml`) | нет |
| `SUBMISSION_PATH` | путь к результату | `/out/submission.csv` (переопределяется в `compose.yaml`) | нет |

Дополнительно поддерживаются `LLM_TIMEOUT`, `LLM_EXTRA_BODY`, `DEV_EMBED_BASE_URL`, `DEV_RERANK_BASE_URL`, `EMBEDDER_API_KEY`, `RERANKER_API_KEY` — см. комментарии в `.env.docker.example`.

### Внешние API

На платформе используются:

| Компонент | Сервис / модель | Переменная | Порядок подключения |
|---|---|---|---|
| LLM | DeepSeek V4.1 Flash | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | адрес и модель — в `.env`, ключ — туда же, в git не коммитим |
| Эмбеддинги | BAAI/bge-m3 | `EMB_MODEL` (и `EMBEDDER_API_KEY`, если ключ отдельный) | `EMB_MODEL=BAAI/bge-m3` |
| Реранкер (необязательно) | Qwen3 Reranker 8B | `RERANKER_MODEL` (и `RERANKER_API_KEY`) | пусто — поиск без реранкера |

Ключей в репозитории нет. Для подключения внешнего API раскомментируйте соответствующий блок в `.env.docker.example` и заполните `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.

### Порты

- `8000` — веб-интерфейс (появится вместе с бэкендом; сервис `api` уже описан в `compose.yaml` под `profiles: ["web"]`).

После появления бэкенда: `docker compose up` → `http://localhost:8000`.

### Зависимости

Все зависимости зафиксированы по версиям в [`requirements.txt`](./requirements.txt). Отдельного списка для Docker нет: локально и в контейнере используется один и тот же набор пакетов.

### Проверка основного сценария

Пакетный прогон:

```bash
docker compose run --rm solution
head -3 out/submission.csv
```

Ожидаемо:

- в `out/submission.csv` есть заголовок `id,answer` и все id из `data/questions.csv`;
- в логе есть `[solution] docs=… chunks=…`, `[solution] индекс: dense=True …` и `[solution] готово: …`;
- нет `Connection refused` (значит, Ollama/внешний API доступны) и нет `PermissionError` (значит, `./out` записывается корректно).

Веб-интерфейс (после появления бэкенда):

```bash
docker compose up
# открыть http://localhost:8000 и задать вопрос
```

### Ollama на хосте

По умолчанию `.env.docker.example` настроен на Ollama на хосте. Внутри контейнера `localhost` — это сам контейнер, поэтому Ollama доступна по имени `host.docker.internal` (в `compose.yaml` уже прописан `extra_hosts: ["host.docker.internal:host-gateway"]` — нужно на Linux, на Windows и macOS имя работает и так).

Скачайте модели:

```bash
ollama pull qwen3:4b     # LLM_MODEL
ollama pull bge-m3       # EMB_MODEL (эмбеддер для индекса)
```

Ollama должна слушать не только `127.0.0.1`, иначе из контейнера до неё не достучаться:

- **Linux (systemd):**
  ```bash
  sudo systemctl edit ollama
  # [Service]
  # Environment="OLLAMA_HOST=0.0.0.0:11434"
  # Environment="OLLAMA_CONTEXT_LENGTH=8192"
  sudo systemctl restart ollama
  ```
- **Windows:** `setx OLLAMA_HOST "0.0.0.0:11434"` и `setx OLLAMA_CONTEXT_LENGTH 8192`, затем полностью закрыть и снова запустить Ollama.
- **macOS:** обычно уже слушает `0.0.0.0`; если нет — `OLLAMA_HOST=0.0.0.0:11434 ollama serve`.

`OLLAMA_CONTEXT_LENGTH=8192` нужен, потому что контекст Ollama по умолчанию 4096 токенов, а промпты в проекте до ~5500.

Проверка связности из контейнера (в образе нет `curl`, используем Python из стандартной библиотеки):

```bash
docker compose run --rm --entrypoint python solution -c \
  "import urllib.request, json; print([m['name'] for m in json.loads(urllib.request.urlopen('http://host.docker.internal:11434/api/tags', timeout=5).read())['models']])"
```

Ожидаемо: список с `qwen3:4b` и `bge-m3`.


### Известные ограничения

- **Индекс строится заново при каждом запуске.** `solution.py` делает вызовы `/embeddings` к `EMB_MODEL`, это занимает время (десятки секунд). Между запусками индекс не кешируется — как и на платформе.
- **Без эмбеддера поиск работает только на BM25.** Если `EMB_MODEL` недоступен, `solution.py` деградирует до BM25 (`dense=False` в логе), прогон не падает.
- **Без реранкера поиск работает без него.** Для локальной Ollama `RERANKER_MODEL` оставляйте пустым — реранкера там нет. При непустом значении и недоступном сервисе реранк отключается после первой ошибки.
- **Ответы на локальной модели хуже, чем на платформенной.** `qwen3:4b` — компромисс по размеру; на платформе используется DeepSeek V4.1 Flash.
- **`TIME_BUDGET_SEC` (по умолчанию 900 с) — бюджет всего прогона.** Если модель медленная (CPU вместо GPU, большой `MAX_TOKENS`), прогон завершится по бюджету и запишет ответы на ту часть вопросов, которую успел. Для полного результата увеличивайте бюджет или уменьшайте `MAX_TOKENS`.
- **Лимиты как на платформе.** У сервиса `solution` в `compose.yaml` заданы `cpus: 8` и `mem_limit: 16g`, чтобы выход за них был виден локально раньше, чем на платформе.
- **Права на `out/` (Linux/WSL).** Контейнер работает от пользователя `LOCAL_UID:LOCAL_GID` (по умолчанию `10001:10001`). Если не делали `export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)` до запуска — прогон может упасть с `PermissionError`; выполните `export` и перезапустите. Альтернатива — один раз поправить владельца: `sudo chown -R "$(id -u):$(id -g)" out`.

### Веб-интерфейс (появится с бэкендом)

Сервис `api` (FastAPI + UI) уже описан в `compose.yaml`, но пока под `profiles: ["web"]` и без каталога `api/`, поэтому по умолчанию не поднимается. После появления бэкенда:

- `docker compose up` — поднимет веб-интерфейс на `http://localhost:8000`;
- `docker compose run --rm solution` — пакетный прогон (останется доступен для файла вопросов);
- healthcheck на `GET /health` уже настроен.

До этого момента для проверки веб-части ничего не нужно — достаточно пакетного прогона.

### Пересборка образа

```bash
docker compose build              # пересобрать образ legal-assistant:local
docker compose build --no-cache   # без кеша (если подозреваете устаревший слой pip install)
```
