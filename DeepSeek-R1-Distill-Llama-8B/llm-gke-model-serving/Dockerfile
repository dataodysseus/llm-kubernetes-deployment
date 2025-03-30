# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
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
    google-cloud-storage

# Set working directory
WORKDIR /app

# Copy the inference script
COPY app.py .

# Set environment variables
ENV MODEL_DIR=/gcs/{{bucket_name}}/model_weights
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Run the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]