# Dockerfile for FastAPI Genie Backend on Cloud Run
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Expose port (Cloud Run uses PORT env var, default 8080)
ENV PORT=8080

# Run the application
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
