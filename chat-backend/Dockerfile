FROM python:3.12-slim

# ── Working directory ──────────────────────────────────────
WORKDIR /app

# ── No system dependencies needed ─────────────────────────
# Unlike the MCP server which uses psycopg2 (C extension),
# this backend is pure Python — no gcc or libpq-dev required.
# Keeps the image lean and the build fast.

# ── Python dependencies ────────────────────────────────────
# Copy requirements first so Docker caches this layer.
# If only backend.py or static/ changes, pip install is skipped.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────
COPY backend.py .

# ── Frontend static files ──────────────────────────────────
# index.html is served by FastAPI's StaticFiles mount at GET /
# Copy from chat-ui/static/ into this image at build time.
# One container serves both the API and the UI.
COPY static/ ./static/

# ── Runtime ────────────────────────────────────────────────
# Cloud Run injects PORT=8080 — uvicorn binds to it.
# Single worker is correct for Cloud Run — scaling happens
# by spawning new container instances, not more workers.
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]