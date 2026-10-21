# Build from the repository root:  docker build -f backend/Dockerfile -t trendybits-api .
FROM python:3.11-slim

ARG INSTALL_ML=false
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRENDYBITS_HOME=/app

WORKDIR /app
COPY backend/requirements.txt backend/requirements-ml.txt backend/
RUN pip install --no-cache-dir -r backend/requirements.txt && \
    if [ "$INSTALL_ML" = "true" ]; then \
      pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r backend/requirements-ml.txt; \
    fi

COPY backend backend
COPY scripts scripts
COPY tests tests
COPY data data
COPY models models

# Make sure datasets, fast models and evaluation telemetry exist inside the image
RUN python scripts/import_ieee_dataport.py --generate >/dev/null && \
    python scripts/train_bert_classifier.py --mode fast >/dev/null

WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
