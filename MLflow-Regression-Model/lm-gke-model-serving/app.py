# app.py - MLflow Model Serving with FastAPI + Gradio
"""
Serve MLflow models from Databricks with FastAPI and Gradio
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import logging
import sys
import os
import pandas as pd
import numpy as np

# Import MLflow model utilities
from model import load_model, get_model_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLflow Model API",
    description="Serving MLflow models from Databricks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
logger.info("Loading MLflow model...")
try:
    model = load_model()
    model_info = get_model_info(model)
    logger.info(f"✅ Model loaded successfully: {model_info}")
except Exception as e:
    logger.error(f"❌ Failed to load model: {str(e)}")
    raise

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "service": "MLflow Model Serving",
        "model_uri": os.getenv("MODEL_URI", "example-model"),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes probes"""
    try:
        # Test model with a simple prediction
        test_data = pd.DataFrame([[1.0]], columns=['x'])
        test_prediction = model.predict(test_data)[0]
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_info": get_model_info(model),
            "test_prediction": float(test_prediction)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/model-info")
async def get_model_information():
    """Get information about the loaded model"""
    try:
        return {
            "model_uri": os.getenv("MODEL_URI", "example-model"),
            "model_info": get_model_info(model),
            "databricks_host": os.getenv("DATABRICKS_HOST", "not-configured")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict")
async def predict_get(x: float):
    """
    GET endpoint for predictions
    Usage: /predict?x=5
    """
    try:
        logger.info(f"Received prediction request: x={x}")
        
        # Create DataFrame for MLflow model (expects column names)
        input_data = pd.DataFrame([[x]], columns=['x'])
        prediction = model.predict(input_data)[0]
        
        result = {
            "input_x": float(x),
            "predicted_y": float(prediction),
            "model_uri": os.getenv("MODEL_URI", "example-model")
        }
        logger.info(f"Prediction result: {result}")
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict")
async def predict_post(data: dict):
    """
    POST endpoint for predictions
    
    Single prediction:
        {"x": 5.0}
    
    Batch prediction:
        {"data": [[1.0], [2.0], [3.0]]}
    """
    try:
        # Handle single value
        if "x" in data:
            x = data.get("x")
            logger.info(f"Received POST prediction request: x={x}")
            input_data = pd.DataFrame([[float(x)]], columns=['x'])
            prediction = model.predict(input_data)[0]
            
            result = {
                "input_x": float(x),
                "predicted_y": float(prediction),
                "model_uri": os.getenv("MODEL_URI", "example-model")
            }
            logger.info(f"Prediction result: {result}")
            return result
        
        # Handle batch predictions
        elif "data" in data:
            input_list = data.get("data")
            logger.info(f"Received batch prediction request: {len(input_list)} items")
            
            # Convert to DataFrame
            input_data = pd.DataFrame(input_list, columns=['x'])
            predictions = model.predict(input_data)
            
            result = {
                "predictions": [float(p) for p in predictions],
                "count": len(predictions),
                "model_uri": os.getenv("MODEL_URI", "example-model")
            }
            logger.info(f"Batch prediction complete: {len(predictions)} predictions")
            return result
        
        else:
            raise HTTPException(status_code=400, detail="Missing 'x' or 'data' parameter")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Gradio UI function
def predict_gradio(x: float) -> float:
    """
    Gradio prediction function
    Takes input x and returns predicted y
    """
    try:
        input_data = pd.DataFrame([[x]], columns=['x'])
        prediction = model.predict(input_data)[0]
        return float(prediction)
    except Exception as e:
        logger.error(f"Gradio prediction error: {str(e)}")
        return f"Error: {str(e)}"

# Create Gradio interface
gradio_interface = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Number(label="Input X", value=5.0),
    outputs=gr.Number(label="Predicted Y"),
    title="MLflow Model from Databricks",
    description=f"Model URI: {os.getenv('MODEL_URI', 'example-model')}",
    examples=[[1], [2], [3], [4], [5], [10], [20]],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, gradio_interface, path="/gradio")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("=" * 50)
    logger.info("MLflow Model Serving Application Started")
    logger.info("=" * 50)
    logger.info(f"Model URI: {os.getenv('MODEL_URI', 'example-model')}")
    logger.info(f"Databricks Host: {os.getenv('DATABRICKS_HOST', 'not-configured')}")
    logger.info(f"FastAPI Docs: http://localhost:8000/docs")
    logger.info(f"Gradio UI: http://localhost:8000/gradio")
    logger.info(f"Health Check: http://localhost:8000/health")
    logger.info(f"Model Info: http://localhost:8000/model-info")
    logger.info("=" * 50)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information"""
    logger.info("Shutting down MLflow Model Serving Application")

# Main execution (for local development)
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )