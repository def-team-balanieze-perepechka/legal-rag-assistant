FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    SUBMISSION_PATH=/out/submission.csv

WORKDIR /app

RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY solution.py ./solution.py
COPY core ./core

RUN mkdir -p /data /out && chown -R app:app /app /data /out

USER app


CMD ["python", "solution.py"]