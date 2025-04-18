# app.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Read token from environment
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not HUGGING_FACE_HUB_TOKEN:
    raise ValueError("HUGGING_FACE_HUB_TOKEN is not set!")

# Load the model and tokenizer
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HUGGING_FACE_HUB_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    token=HUGGING_FACE_HUB_TOKEN
)

# Define request format
class MessagesRequest(BaseModel):
    messages: list
    
@app.post("/generate")
async def generate(request: MessagesRequest):
    try:
        # Format the messages using the tokenizer's chat template
        prompt = tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate the response
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=512,  # Adjust as needed
            temperature=0.7,      # Lower values make the output more deterministic
            top_k=50,             # Lower k focuses on higher probability tokens
            top_p=0.95,           # Lower values make the output more focused
            do_sample=True        # Enable sampling
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        
        # Decode the response
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract only the assistant's response (remove the prompt)
        # assistant_response = response.split(prompt)[-1].strip()
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
