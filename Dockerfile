FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Install the CPU wheel first to avoid downloading CUDA dependencies.
COPY requirements-lock.txt ./
RUN python -m pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements-lock.txt \
    && python -m pip check

COPY flyheal ./flyheal
COPY configs ./configs
COPY tests ./tests
COPY pyproject.toml LICENSE THIRD_PARTY_NOTICES.md ./
RUN mkdir -p results

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/runs', timeout=3).read()"

CMD ["python", "-m", "flyheal.ui", "--host", "0.0.0.0", "--port", "8765", "--no-open"]
