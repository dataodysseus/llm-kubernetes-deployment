from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.model import load_model
import gradio as gr

app = FastAPI(title="Linear Regression API + Gradio UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()

@app.get("/predict")
def predict(x: float):
    y = model.predict([[x]])[0]
    return {"input_x": x, "predicted_y": y}

def predict_ui(x):
    return model.predict([[x]])[0]

gradio_app = gr.Interface(fn=predict_ui, inputs="number", outputs="number")
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

