# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch \
    transformers \
    uvicorn \
    fastapi \
    pydantic \
    accelerate \
    sentencepiece \
    bitsandbytes \
    protobuf==3.20.*

# Copy the inference script
COPY app.py /app.py

# Set environment variables
ENV MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Run the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]