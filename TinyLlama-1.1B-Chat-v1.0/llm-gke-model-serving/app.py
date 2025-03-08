# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Define a request model for the prompt
class MessagesRequest(BaseModel):
    messages: list  # List of dictionaries with "role" and "content" keys

# Load the model and tokenizer
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")

@app.post("/generate")
async def generate(request: MessagesRequest):
    try:
        # Format the messages using the tokenizer's chat template
        prompt = tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate the response
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,  # Adjust as needed
            temperature=0.7,      # Lower values make the output more deterministic
            top_k=50,             # Lower k focuses on higher probability tokens
            top_p=0.95,           # Lower values make the output more focused
            do_sample=True        # Enable sampling
        )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response (remove the prompt)
        assistant_response = response.split(prompt)[-1].strip()
        
        return {"response": assistant_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))