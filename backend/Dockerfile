FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRENDYBITS_HOME=/app

WORKDIR /app

# Copy requirement files first to leverage Docker layer caching
COPY backend/requirements.txt backend/requirements-ml.txt backend/

# Install standard and ML dependencies unconditionally for inference
RUN pip install --no-cache-dir -r backend/requirements.txt && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r backend/requirements-ml.txt

# Copy only what is strictly necessary for runtime
COPY backend backend
COPY data data
COPY models models

WORKDIR /app/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]