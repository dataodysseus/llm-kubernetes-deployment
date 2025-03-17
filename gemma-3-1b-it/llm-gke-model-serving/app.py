# app.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM
import torch

app = FastAPI()

# Read token from environment
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not HUGGING_FACE_HUB_TOKEN:
    raise ValueError("HUGGING_FACE_HUB_TOKEN is not set!")

# Load the model and tokenizer
MODEL_ID = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HUGGING_FACE_HUB_TOKEN)
model = Gemma3ForCausalLM.from_pretrained(
    MODEL_ID, 
    token=HUGGING_FACE_HUB_TOKEN
).eval()

# Define request format
class MessagesRequest(BaseModel):
    messages: list
    
@app.post("/generate")
async def generate(request: MessagesRequest):
    try:
        # Tokenize the prompt
        inputs = tokenizer.apply_chat_template(
            request.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate the response
        with torch.inference_mode():
          outputs = model.generate(**inputs, max_new_tokens=384)

        # Decode the response
        response = tokenizer.batch_decode(outputs)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))