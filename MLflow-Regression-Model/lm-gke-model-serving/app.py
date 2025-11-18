# app.py - MLflow Model Serving (Simplified)
"""
Load and serve MLflow models from Databricks directly in app.py
Similar pattern to Hugging Face model loading
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gradio as gr
import mlflow
import mlflow.pyfunc
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="MLflow Model API",
    description="Serving MLflow models from Databricks"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read environment variables
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
MODEL_URI = os.getenv("MODEL_URI")  # e.g., "models:/my-model/Production" or "runs:/run-id/model"

# Validate required environment variables
if not DATABRICKS_HOST:
    raise ValueError("DATABRICKS_HOST is not set!")
if not DATABRICKS_TOKEN:
    raise ValueError("DATABRICKS_TOKEN is not set!")
if not MODEL_URI:
    raise ValueError("MODEL_URI is not set!")

# Configure Databricks connection
logger.info(f"Configuring Databricks connection: {DATABRICKS_HOST}")
mlflow.set_tracking_uri("databricks")
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

# Load the model
logger.info(f"Loading model from: {MODEL_URI}")
model = mlflow.pyfunc.load_model(MODEL_URI)
logger.info("✅ Model loaded successfully!")

# Define request format
class PredictionRequest(BaseModel):
    x: float

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "MLflow Model Serving",
        "model_uri": MODEL_URI
    }

@app.get("/health")
async def health_check():
    """Health check for Kubernetes"""
    try:
        # Test prediction
        test_data = pd.DataFrame([[1.0]], columns=['x'])
        test_prediction = model.predict(test_data)[0]
        return {
            "status": "healthy",
            "model_loaded": True,
            "test_prediction": float(test_prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/predict")
async def predict_get(x: float):
    """GET endpoint: /predict?x=5"""
    try:
        logger.info(f"Prediction request: x={x}")
        input_data = pd.DataFrame([[x]], columns=['x'])
        prediction = model.predict(input_data)[0]
        
        return {
            "input_x": float(x),
            "predicted_y": float(prediction),
            "model_uri": MODEL_URI
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_post(request: PredictionRequest):
    """POST endpoint with JSON body"""
    try:
        logger.info(f"Prediction request: x={request.x}")
        input_data = pd.DataFrame([[request.x]], columns=['x'])
        prediction = model.predict(input_data)[0]
        
        return {
            "input_x": float(request.x),
            "predicted_y": float(prediction),
            "model_uri": MODEL_URI
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Gradio UI
def predict_gradio(x: float) -> float:
    """Gradio prediction function"""
    try:
        input_data = pd.DataFrame([[x]], columns=['x'])
        prediction = model.predict(input_data)[0]
        return float(prediction)
    except Exception as e:
        return f"Error: {str(e)}"

gradio_interface = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Number(label="Input X", value=5.0),
    outputs=gr.Number(label="Predicted Y"),
    title="MLflow Model from Databricks",
    description=f"Model URI: {MODEL_URI}",
    examples=[[1], [2], [3], [4], [5], [10]],
    theme=gr.themes.Soft()
)

# Mount Gradio to FastAPI
app = gr.mount_gradio_app(app, gradio_interface, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )