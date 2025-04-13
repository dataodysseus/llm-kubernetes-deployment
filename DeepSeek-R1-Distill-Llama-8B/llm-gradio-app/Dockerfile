# Dockerfile for Gradio App
FROM python:3.9-slim

# Install dependencies
RUN pip install --no-cache-dir gradio requests

# Copy the Gradio app code
COPY app.py /app.py

# Expose the port
EXPOSE 7860

ENV LLM_SERVICE_URL="http://llm-service.gke-ai-namespace.svc.cluster.local"

# Run the app
CMD ["python", "/app.py"]


