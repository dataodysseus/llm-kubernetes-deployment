# Dockerfile for Gradio App
FROM python:3.9-slim

# Install dependencies
# Install stable versions that work together
RUN pip install --no-cache-dir \
    gradio==3.50.2 \
    huggingface_hub==0.22.2 \
    requests


# Copy the Gradio app code
COPY app.py /app.py

# Expose the port
EXPOSE 7860

# Run the app
CMD ["python", "/app.py"]
