# model.py - ML Model Training and Loading
"""
Linear Regression Model utilities
Handles model training, saving, and loading
"""

from sklearn.linear_model import LinearRegression
import joblib
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# Model path - will be created in the same directory as app.py
MODEL_PATH = "model.pkl"

def train_model():
    """
    Train a simple linear regression model
    Training data follows: y = 2x + 1
    """
    try:
        logger.info("Training new model...")
        
        # Sample training data (x → y)
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([3, 5, 7, 9, 11])  # y = 2x + 1
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Save model
        joblib.dump(model, MODEL_PATH)
        logger.info(f"✅ Model trained and saved at {MODEL_PATH}")
        logger.info(f"   Coefficients: {model.coef_}")
        logger.info(f"   Intercept: {model.intercept_}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {str(e)}")
        raise

def load_model():
    """
    Load the trained model
    If model doesn't exist, train a new one
    """
    try:
        if not os.path.exists(MODEL_PATH):
            logger.info(f"Model file not found at {MODEL_PATH}. Training new model...")
            return train_model()
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        logger.info("✅ Model loaded successfully")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise

def get_model_info():
    """
    Get information about the current model
    """
    try:
        model = load_model()
        return {
            "model_type": "LinearRegression",
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "equation": f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}"
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # For testing purposes
    logging.basicConfig(level=logging.INFO)
    
    print("Testing model training and loading...")
    model = load_model()
    
    # Test predictions
    test_values = [1, 2, 3, 4, 5, 10]
    print("\nTest Predictions:")
    print("-" * 40)
    for x in test_values:
        y = model.predict([[x]])[0]
        print(f"x = {x:2d}  →  y = {y:.2f}")
    
    print("\nModel Info:")
    print("-" * 40)
    info = get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")
