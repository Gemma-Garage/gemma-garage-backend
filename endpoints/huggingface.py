import os
import asyncio
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import httpx
from google.cloud import firestore
from typing import Optional
import json
from huggingface_hub import HfApi, upload_file, create_repo, attach_huggingface_oauth, parse_huggingface_oauth
from config_vars import NEW_DATA_BUCKET

router = APIRouter()

# Initialize Firestore client
db = firestore.Client()

class HFUploadRequest(BaseModel):
    user_id: str
    model_name: str
    request_id: str
    description: Optional[str] = "Fine-tuned model from Gemma Garage"
    private: bool = False

class HFInferenceRequest(BaseModel):
    model_name: str
    prompt: str
    max_new_tokens: int = 100

@router.post("/upload_model")
async def upload_model_to_hf(request: HFUploadRequest, fastapi_request: Request):
    """Upload a fine-tuned model to Hugging Face."""
    
    # Parse OAuth info from the request
    oauth_info = parse_huggingface_oauth(fastapi_request)
    if oauth_info is None:
        raise HTTPException(status_code=401, detail="Please log in with Hugging Face first")
    
    # Get access token from OAuth info
    hf_token = oauth_info.access_token
    hf_username = oauth_info.user_info.preferred_username
    
    # Initialize HF API
    api = HfApi(token=hf_token)
    
    # Create repository name
    repo_name = f"{hf_username}/{request.model_name}"
    
    try:
        # Create repository on Hugging Face
        repo_url = create_repo(
            repo_id=repo_name,
            token=hf_token,
            private=request.private,
            exist_ok=True
        )
        
        # TODO: Download model files from GCS and upload to HF
        # This is a placeholder - in practice, you'd:
        # 1. Download model files from GCS bucket using request.request_id
        # 2. Upload each file to the HF repository
        # 3. Create model card with description
        
        # For now, return success with repository URL
        return {
            "success": True,
            "repo_url": f"https://huggingface.co/{repo_name}",
            "message": f"Model repository created: {repo_name}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {str(e)}")

@router.post("/inference")
async def hf_inference(request: HFInferenceRequest, fastapi_request: Request):
    """Run inference using Hugging Face Inference API."""
    
    # Parse OAuth info from the request
    oauth_info = parse_huggingface_oauth(fastapi_request)
    if oauth_info is None:
        raise HTTPException(status_code=401, detail="Please log in with Hugging Face first")
    
    # Get access token from OAuth info
    hf_token = oauth_info.access_token
    
    # Call Hugging Face Inference API
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api-inference.huggingface.co/models/{request.model_name}",
            headers={
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": request.prompt,
                "parameters": {
                    "max_new_tokens": request.max_new_tokens,
                    "return_full_text": False
                }
            },
            timeout=30.0
        )
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, 
            detail=f"Hugging Face API error: {response.text}"
        )
    
    result = response.json()
    
    # Extract generated text
    if isinstance(result, list) and len(result) > 0:
        generated_text = result[0].get("generated_text", "")
    else:
        generated_text = str(result)
    
    return {
        "response": generated_text,
        "model": request.model_name
    }

@router.get("/status")
async def get_hf_connection_status(request: Request):
    """Get Hugging Face connection status for the current user."""
    
    oauth_info = parse_huggingface_oauth(request)
    
    if oauth_info is None:
        return {
            "connected": False,
            "username": None,
            "user_info": None
        }
    
    return {
        "connected": True,
        "username": oauth_info.user_info.preferred_username,
        "user_info": {
            "name": oauth_info.user_info.name,
            "email": oauth_info.user_info.email,
            "picture": oauth_info.user_info.picture,
            "is_pro": oauth_info.user_info.is_pro
        },
        "expires_at": oauth_info.access_token_expires_at.isoformat()
    }
