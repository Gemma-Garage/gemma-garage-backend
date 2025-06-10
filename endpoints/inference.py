import os
from fastapi import FastAPI, Request, APIRouter
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from google.cloud import storage

router = APIRouter()

MODEL_DIR = "/app/model"
GCS_BUCKET = "llm-garage-models"
GCS_PREFIX = "gemma-peft-vertex-output/model"

class PromptRequest(BaseModel):
    prompt: str
    request_id: str

def download_model_from_gcs(bucket_name, prefix, local_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    os.makedirs(local_dir, exist_ok=True)
    for blob in blobs:
        file_path = os.path.join(local_dir, blob.name.replace(prefix + "/", ""))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        blob.download_to_filename(file_path)

# Download model at startup

@router.post("/predict")
def predict(request: PromptRequest):
    request_id = request.request_id
    model_path = f"{GCS_PREFIX}/{request_id}/final_model/"
    if not model_path:
        model_path = MODEL_DIR
    download_model_from_gcs(GCS_BUCKET, GCS_PREFIX, MODEL_DIR)
    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}
