# model.py - MLflow Model Loading from Databricks
"""
Load and serve ML models from Databricks MLflow
Supports both Model Registry and run-based models
"""

import mlflow
import mlflow.pyfunc
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration - Set these via environment variables
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")  # e.g., https://adb-xxxxx.azuredatabricks.net
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")  # Personal Access Token
MODEL_URI = os.getenv("MODEL_URI")  # e.g., "models:/my-model/production" or "runs:/run-id/model"

def configure_databricks():
    """Configure MLflow to connect to Databricks"""
    if DATABRICKS_HOST and DATABRICKS_TOKEN:
        mlflow.set_tracking_uri("databricks")
        os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
        os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
        logger.info(f"✅ Configured Databricks connection: {DATABRICKS_HOST}")
    else:
        logger.warning("⚠️ DATABRICKS_HOST or DATABRICKS_TOKEN not set")

def load_model_from_registry(model_name: str, stage: str = "Production"):
    """
    Load model from Databricks Model Registry
    
    Args:
        model_name: Name of the registered model (e.g., "linear_regression_model")
        stage: Model stage - "Production", "Staging", "Archived", or version number
    
    Returns:
        Loaded MLflow model
    """
    try:
        configure_databricks()
        
        # Model URI for registry
        if stage.isdigit():
            model_uri = f"models:/{model_name}/{stage}"  # Specific version
        else:
            model_uri = f"models:/{model_name}/{stage}"  # Stage alias
        
        logger.info(f"Loading model from registry: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"✅ Model loaded successfully from registry")
        
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model from registry: {str(e)}")
        raise

def load_model_from_run(run_id: str, artifact_path: str = "model"):
    """
    Load model from a specific MLflow run
    
    Args:
        run_id: MLflow run ID (e.g., "abc123def456...")
        artifact_path: Path to model artifact (default: "model")
    
    Returns:
        Loaded MLflow model
    """
    try:
        configure_databricks()
        
        model_uri = f"runs:/{run_id}/{artifact_path}"
        logger.info(f"Loading model from run: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"✅ Model loaded successfully from run")
        
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model from run: {str(e)}")
        raise

def load_model_from_uri(model_uri: str):
    """
    Load model from any valid MLflow URI
    
    Supported formats:
    - "models:/model-name/Production"
    - "models:/model-name/Staging"
    - "models:/model-name/1" (version number)
    - "runs:/run-id/model"
    - "dbfs:/path/to/model"
    
    Args:
        model_uri: MLflow model URI
    
    Returns:
        Loaded MLflow model
    """
    try:
        configure_databricks()
        
        logger.info(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"✅ Model loaded successfully")
        
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model from URI: {str(e)}")
        raise

def load_model():
    """
    Main function to load model based on MODEL_URI environment variable
    Falls back to a default model if not configured
    """
    try:
        if MODEL_URI:
            return load_model_from_uri(MODEL_URI)
        else:
            logger.warning("⚠️ MODEL_URI not set, using example model")
            return load_example_model()
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        logger.info("Falling back to example model...")
        return load_example_model()

def load_example_model():
    """
    Load a simple example model for testing
    Used when Databricks connection is not available
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    logger.info("Creating example linear regression model...")
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])
    
    model = LinearRegression()
    model.fit(X, y)
    
    logger.info("✅ Example model created")
    return model

def get_model_info(model):
    """
    Get information about the loaded model
    """
    try:
        info = {
            "model_type": str(type(model)),
            "mlflow_model": isinstance(model, mlflow.pyfunc.PyFuncModel)
        }
        
        # Try to get MLflow metadata
        if hasattr(model, 'metadata'):
            info["metadata"] = model.metadata
        
        return info
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("MLflow Model Loading Examples:")
    print("=" * 50)
    
    # Example 1: Load from registry (production)
    print("\n1. Load from Model Registry (Production):")
    print("   MODEL_URI='models:/my-model/Production'")
    
    # Example 2: Load from registry (specific version)
    print("\n2. Load from Model Registry (Version 3):")
    print("   MODEL_URI='models:/my-model/3'")
    
    # Example 3: Load from run
    print("\n3. Load from specific run:")
    print("   MODEL_URI='runs:/abc123def456/model'")
    
    # Example 4: Load from DBFS
    print("\n4. Load from DBFS path:")
    print("   MODEL_URI='dbfs:/models/my-model'")
    
    print("\n" + "=" * 50)
    print("Testing with example model...")
    model = load_model()
    print(f"Model loaded: {type(model)}")