FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY benchmarks ./benchmarks
COPY README.md LICENSE ./

RUN pip install --no-cache-dir -e .

# Smoke benchmark needs no API key and runs in under 10s. Useful as a healthcheck.
HEALTHCHECK --interval=30s --timeout=15s --start-period=5s --retries=2 \
    CMD python -c "from multi_agent_supervisor import build_supervisor; print('ok')" || exit 1

ENTRYPOINT ["python", "-m", "multi_agent_supervisor.cli"]
CMD ["--help"]
