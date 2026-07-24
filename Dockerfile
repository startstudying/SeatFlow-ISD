FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[api,tracking]"

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=12 --start-period=20s \
  CMD python -c "import json,urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)); raise SystemExit(0 if data.get('status') == 'ok' else 1)"

CMD ["sh", "-c", "python -m seatflow serve-api --config /workspace/configs/default.toml --provider ${SEATFLOW_DATA_PROVIDER:-sample} --data-root /workspace/data --host 0.0.0.0 --port 8000"]
