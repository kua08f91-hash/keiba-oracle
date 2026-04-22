FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy application code
COPY backend/ backend/
COPY data/ data/

# Create data directory for SQLite fallback
RUN mkdir -p data

# Default port (Railway sets PORT env var)
ENV PORT=8000

EXPOSE $PORT

CMD gunicorn backend.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
