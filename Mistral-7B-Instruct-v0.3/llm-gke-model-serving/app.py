# # app.py
# import os
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# app = FastAPI()

# # Read token from environment
# HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# if not HUGGING_FACE_HUB_TOKEN:
#     raise ValueError("HUGGING_FACE_HUB_TOKEN is not set!")

# # Load the model and tokenizer
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HUGGING_FACE_HUB_TOKEN)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID, 
#     torch_dtype=torch.float16, 
#     device_map="auto", 
#     token=HUGGING_FACE_HUB_TOKEN
# )

# # Define request format
# class MessagesRequest(BaseModel):
#     messages: list
    
# @app.post("/generate")
# async def generate(request: MessagesRequest):
#     try:
#         # Format the messages using the tokenizer's chat template
#         prompt = tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True)
        
#         # Tokenize the prompt
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
#         # Generate the response
#         generated_ids = model.generate(
#             inputs.input_ids,
#             max_new_tokens=512,  # Adjust as needed
#             temperature=0.7,      # Lower values make the output more deterministic
#             top_k=50,             # Lower k focuses on higher probability tokens
#             top_p=0.95,           # Lower values make the output more focused
#             do_sample=True        # Enable sampling
#         )

#         generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        
#         # Decode the response
#         response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
#         # Extract only the assistant's response (remove the prompt)
#         # assistant_response = response.split(prompt)[-1].strip()
        
#         return {"response": response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# ----
# app.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from google.cloud import storage

app = FastAPI()

# Read token from environment
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

if not HUGGING_FACE_HUB_TOKEN:
    raise ValueError("HUGGING_FACE_HUB_TOKEN is not set!")

# GCS bucket and model path
GCS_BUCKET_NAME = "llm-test-bucket-2025"
MODEL_PATH_IN_GCS = "model_weights"
LOCAL_MODEL_DIR = "/tmp/model_weights"

# Download model from GCS to local directory
def download_model_from_gcs(bucket_name, source_blob_name, destination_dir):
    """Downloads a model from GCS to a local directory."""
    os.makedirs(destination_dir, exist_ok=True)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_name)
    
    for blob in blobs:
        # Construct the local file path
        local_file_path = os.path.join(destination_dir, os.path.basename(blob.name))
        # Download the file
        blob.download_to_filename(local_file_path)
        print(f"Downloaded {blob.name} to {local_file_path}")

# Download the model weights from GCS
download_model_from_gcs(GCS_BUCKET_NAME, MODEL_PATH_IN_GCS, LOCAL_MODEL_DIR)

# Load the tokenizer and model from the local directory
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HUGGING_FACE_HUB_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR, 
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