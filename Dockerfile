FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    SUBMISSION_PATH=/out/submission.csv

WORKDIR /app

RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

ARG REQS=requirements.txt

COPY --chown=app:app requirements.txt requirements-api.txt requirements-ui.txt ./

RUN pip install --no-cache-dir -r "${REQS}"

COPY --chown=app:app solution.py ./
COPY --chown=app:app core ./core
COPY --chown=app:app api ./api
COPY --chown=app:app ui ./ui

USER app

EXPOSE 8000 8501

CMD ["python", "solution.py"]