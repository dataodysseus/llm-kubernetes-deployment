# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.model import load_model

app = FastAPI(title="Linear Regression Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()  # <- ensure your load_model() returns a trained model

@app.get("/")
def health_check():
    return {"status": "OK", "message": "Model endpoint is running"}

@app.post("/predict")
def predict(features: dict):
    """
    Expecting payload like:
    {
      "x": 5.2
    }
    """
    x = features.get("x")
    if x is None:
        return {"error": "Please provide feature value 'x'"}

    prediction = model.predict([[x]])  # Adjust based on your model type
    return {"prediction": prediction[0]}


