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
    git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3 \
    uvicorn \
    fastapi \
    pydantic \
    accelerate 

# Copy the inference script
COPY app.py /app.py

# Set environment variables
ENV MODEL_ID=google/gemma-3-1b-it
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Run the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]