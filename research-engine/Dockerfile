FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/seatflow-venv/bin:$PATH

WORKDIR /build

RUN python -m venv /opt/seatflow-venv

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install .


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/seatflow-venv/bin:$PATH

RUN groupadd --system --gid 10001 seatflow \
    && useradd --system --uid 10001 --gid seatflow --home-dir /app --create-home seatflow

COPY --from=builder /opt/seatflow-venv /opt/seatflow-venv

WORKDIR /app
USER seatflow:seatflow

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=8 --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/research/v1/health', timeout=3).read()"

CMD ["uvicorn", "seatflow.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
