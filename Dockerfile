FROM python:3.10-slim

WORKDIR /app

# Copy and install dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend and root-level dependencies
COPY backend/ ./backend/
COPY models/ ./models/
COPY scripts/ ./scripts/

WORKDIR /app/backend
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]