# Dockerfile for Gradio App
FROM python:3.10-slim

# Install dependencies
RUN pip install --no-cache-dir \
    gradio==4.8.0 \
    requests

# Copy the Gradio app code
COPY app.py /app.py

# Expose the port
EXPOSE 7860

# Run the app
CMD ["python", "/app.py"]
