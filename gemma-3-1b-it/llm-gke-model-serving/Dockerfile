# Use slim Python image for smaller footprint
FROM python:3.9-slim

# Install system dependencies (minimal to reduce image size)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first for caching benefits
COPY requirements.txt /requirements.txt

# Install Python dependencies in one step to leverage Docker caching
RUN pip install --no-cache-dir -r /requirements.txt

# Copy the inference script
COPY app.py /app.py

# Set environment variables
ENV MODEL_ID=google/gemma-3-1b-it
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8080

# Run the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
