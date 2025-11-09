# app/model.py
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np
import os

MODEL_PATH = "app/model.pkl"

def train_model():
    # Sample training data (x → y)
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])  # y = 2x + 1

    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved at {MODEL_PATH}")

def load_model():
    if not os.path.exists(MODEL_PATH):
        train_model()
    return joblib.load(MODEL_PATH)
