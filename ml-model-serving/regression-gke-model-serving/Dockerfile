# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    gradio \
    scikit-learn \
    joblib \
    uvicorn \
    numpy

# Copy the application files
COPY app.py /app.py
COPY model.py /model.py

# Set environment variables
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
