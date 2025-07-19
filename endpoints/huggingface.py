import os
import asyncio
import tempfile
import shutil
import time
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import httpx
from google.cloud import firestore, storage
from typing import Optional
import json
import secrets
import urllib.parse
from datetime import datetime, timedelta
from huggingface_hub import HfApi, upload_file, upload_folder, create_repo, parse_huggingface_oauth
from config_vars import NEW_DATA_BUCKET

router = APIRouter()

# Initialize Firestore client
db = firestore.Client()

# OAuth configuration
HUGGINGFACE_CLIENT_ID = os.getenv("HUGGINGFACE_CLIENT_ID")
HUGGINGFACE_CLIENT_SECRET = os.getenv("HUGGINGFACE_CLIENT_SECRET")
# Fix: Use the production backend URL for redirect URI
HUGGINGFACE_REDIRECT_URI = os.getenv("HUGGINGFACE_REDIRECT_URI", "https://llm-garage-513913820596.us-central1.run.app/oauth/huggingface/callback")
# Fix: Use the production frontend URL
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://gemma-garage.web.app")

# Rate limiting tracking
api_call_history = {}
MAX_CALLS_PER_MINUTE = 10  # Conservative limit for HF OAuth endpoints
RATE_LIMIT_WINDOW = 60  # seconds

def check_rate_limit(endpoint: str) -> bool:
    """Check if we're within rate limits for a specific endpoint."""
    current_time = time.time()
    
    if endpoint not in api_call_history:
        api_call_history[endpoint] = []
    
    # Remove calls older than the window
    api_call_history[endpoint] = [
        call_time for call_time in api_call_history[endpoint]
        if current_time - call_time < RATE_LIMIT_WINDOW
    ]
    
    # Check if we're under the limit
    if len(api_call_history[endpoint]) >= MAX_CALLS_PER_MINUTE:
        return False
    
    # Record this call
    api_call_history[endpoint].append(current_time)
    return True

async def make_hf_request_with_retry(client: httpx.AsyncClient, method: str, url: str, **kwargs):
    """Make HTTP request to HF with basic error handling - NO RETRIES for rate limits."""
    try:
        if method.upper() == "POST":
            response = await client.post(url, **kwargs)
        else:
            response = await client.get(url, **kwargs)
        
        # Log rate limiting but don't retry - that makes it worse!
        if response.status_code == 429:
            print(f"Rate limited by HF on {url} - NOT retrying to avoid making it worse")
        
        return response
        
    except Exception as e:
        print(f"Request to {url} failed with exception: {e}")
        raise e

class HFUploadRequest(BaseModel):
    user_id: Optional[str] = None
    model_name: str
    request_id: str
    description: Optional[str] = "Fine-tuned model from Gemma Garage"
    private: bool = False
    base_model: Optional[str] = "google/gemma-2b"

class HFInferenceRequest(BaseModel):
    model_name: str
    prompt: str
    max_new_tokens: int = 100

@router.get("/config")
async def check_oauth_config():
    """Check OAuth configuration status."""
    return {
        "client_id_configured": bool(HUGGINGFACE_CLIENT_ID),
        "client_secret_configured": bool(HUGGINGFACE_CLIENT_SECRET),
        "redirect_uri": HUGGINGFACE_REDIRECT_URI,
        "client_id_preview": HUGGINGFACE_CLIENT_ID[:8] + "..." if HUGGINGFACE_CLIENT_ID else None
    }

@router.get("/rate-limit-status")
async def get_rate_limit_status():
    """Get current rate limit status for HF API calls."""
    current_time = time.time()
    status = {}
    
    for endpoint, calls in api_call_history.items():
        # Filter recent calls
        recent_calls = [call for call in calls if current_time - call < RATE_LIMIT_WINDOW]
        remaining = max(0, MAX_CALLS_PER_MINUTE - len(recent_calls))
        
        status[endpoint] = {
            "calls_in_last_minute": len(recent_calls),
            "limit": MAX_CALLS_PER_MINUTE,
            "remaining": remaining,
            "can_make_request": remaining > 0
        }
    
    return {
        "rate_limits": status,
        "window_seconds": RATE_LIMIT_WINDOW,
        "max_calls_per_minute": MAX_CALLS_PER_MINUTE
    }

def sanitize_username(username: str) -> str:
    """Sanitize username for HuggingFace repository naming."""
    if not username:
        return "huggingface-user"
    
    # Replace spaces and invalid characters with hyphens
    sanitized = username.replace(" ", "-").replace("_", "-")
    # Remove any characters that aren't alphanumeric or hyphens
    sanitized = "".join(c for c in sanitized if c.isalnum() or c == "-")
    # Remove consecutive hyphens
    sanitized = "-".join(part for part in sanitized.split("-") if part)
    # Ensure it doesn't start or end with hyphens
    sanitized = sanitized.strip("-")
    # Convert to lowercase
    sanitized = sanitized.lower()
    
    # Ensure it's not empty and has reasonable length
    if not sanitized or len(sanitized) < 1:
        return "huggingface-user"
    if len(sanitized) > 30:  # Keep reasonable length
        sanitized = sanitized[:30]
    
    return sanitized

def get_oauth_info(request: Request):
    """Get OAuth info using the official Hugging Face OAuth library."""
    try:
        oauth_info = parse_huggingface_oauth(request)
        if oauth_info is None:
            return None
        
        # Convert to our expected format
        return {
            "access_token": oauth_info.access_token,
            "user_info": {
                "name": oauth_info.user_info.preferred_username,
                "email": oauth_info.user_info.email,
                "picture": oauth_info.user_info.picture,
                "isPro": oauth_info.user_info.is_pro,
                "sub": oauth_info.user_info.sub
            },
            "expires_at": oauth_info.access_token_expires_at,
            "scope": oauth_info.scope
        }
    except Exception as e:
        print(f"Error parsing OAuth info: {e}")
        return None

@router.post("/upload_model")
async def upload_model_to_hf(request: HFUploadRequest, fastapi_request: Request):
    """Upload a fine-tuned model to Hugging Face."""
    
    # Get OAuth info using official library
    oauth_data = get_oauth_info(fastapi_request)
    if not oauth_data:
        raise HTTPException(status_code=401, detail="Please log in with Hugging Face first")
    
    hf_token = oauth_data["access_token"]
    user_info = oauth_data["user_info"]
    
    hf_username = sanitize_username(user_info["name"])
    
    # Initialize HF API
    api = HfApi(token=hf_token)
    
    # Create repository name
    repo_name = f"{hf_username}/{request.model_name}"
    
    try:
        # First, check token permissions by getting user info
        try:
            user_info = api.whoami()
            print(f"Token permissions check - User: {user_info}")
            
            # Check if user has necessary permissions
            if not user_info.get("auth", {}).get("accessToken", {}).get("role") in ["write", "admin"]:
                print(f"User permissions: {user_info.get('auth', {})}")
        except Exception as perm_error:
            print(f"Permission check failed: {perm_error}")
            # Continue anyway, let the actual operation fail with a more specific error
        
        # Create repository on Hugging Face
        print(f"Attempting to create repository: {repo_name}")
        repo_url = create_repo(
            repo_id=repo_name,
            token=hf_token,
            private=request.private,
            exist_ok=True
        )
        print(f"Repository created successfully: {repo_url}")
        
        # Download model files from GCS and upload to HF
        base_gcs_output_bucket_uri = os.environ.get("NEW_MODEL_OUTPUT_BUCKET")
        if not base_gcs_output_bucket_uri:
            raise HTTPException(status_code=500, detail="NEW_MODEL_OUTPUT_BUCKET environment variable not set")

        gcs_model_directory_path = f"{base_gcs_output_bucket_uri.rstrip('/')}/model/{request.request_id}/final_model/"
        
        # Parse GCS path
        if not gcs_model_directory_path.startswith("gs://"):
            raise ValueError("GCS path must start with gs://")
        
        path_parts = gcs_model_directory_path[5:].split("/", 1)
        if len(path_parts) != 2:
            raise ValueError("Invalid GCS path format")
        
        bucket_name, model_directory_prefix = path_parts
        
        # Initialize Google Cloud Storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # List all model files
        blobs = list(bucket.list_blobs(prefix=model_directory_prefix))
        if not blobs:
            raise HTTPException(
                status_code=404, 
                detail=f"No model files found in GCS path {gcs_model_directory_path}"
            )
        
        # Create temporary directory to download model files
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Downloading model files from GCS to {temp_dir}")
            
            # Download all model files to temporary directory
            for blob in blobs:
                if blob.name.endswith('/'):
                    continue  # Skip directory markers
                
                # Get the relative path within the model directory
                relative_path = blob.name[len(model_directory_prefix):].lstrip('/')
                if not relative_path:
                    continue
                
                local_file_path = os.path.join(temp_dir, relative_path)
                
                # Create directory if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Download file
                print(f"Downloading {blob.name} to {local_file_path}")
                blob.download_to_filename(local_file_path)
            
            # Create model card
            model_card_content = f"""---
language: en
license: apache-2.0
tags:
- fine-tuned
- gemma
- lora
- gemma-garage
base_model: {request.base_model}
pipeline_tag: text-generation
---

# {request.model_name}

{request.description}

This model was fine-tuned using [Gemma Garage](https://github.com/your-repo/gemma-garage), a platform for fine-tuning Gemma models with LoRA.

## Model Details

- **Base Model**: {request.base_model}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Platform**: Gemma Garage
- **Fine-tuned on**: {datetime.now().strftime('%Y-%m-%d')}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = AutoModelForCausalLM.from_pretrained("{repo_name}")

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

This model was fine-tuned using the Gemma Garage platform with the following configuration:
- Request ID: {request.request_id}
- Training completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

For more information about Gemma Garage, visit [our GitHub repository](https://github.com/your-repo/gemma-garage).
"""
            
            # Save model card
            model_card_path = os.path.join(temp_dir, "README.md")
            with open(model_card_path, "w", encoding="utf-8") as f:
                f.write(model_card_content)
            
            print(f"Uploading model files from {temp_dir} to HF repo {repo_name}")
            
            # Upload all files to HuggingFace repository
            upload_folder(
                folder_path=temp_dir,
                repo_id=repo_name,
                token=hf_token,
                commit_message=f"Upload fine-tuned model from Gemma Garage (request: {request.request_id})"
            )
        
        return {
            "success": True,
            "repo_url": f"https://huggingface.co/{repo_name}",
            "repo_name": repo_name,
            "message": f"Model successfully uploaded to {repo_name}",
            "files_uploaded": len([b for b in blobs if not b.name.endswith('/')])
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error uploading model to HF: {error_msg}")
        
        # Provide more specific error messages for common issues
        if "403 Forbidden" in error_msg and "don't have the rights" in error_msg:
            detailed_error = (
                f"Permission denied: You don't have the rights to create models under the namespace '{hf_username}'. "
                "This could be due to:\n"
                "1. Your Hugging Face token doesn't have 'write-repos' permissions\n"
                "2. Your account may need to be verified or upgraded\n"
                "3. Try creating the repository manually first on HuggingFace.co\n"
                f"Original error: {error_msg}"
            )
        elif "401" in error_msg or "authentication" in error_msg.lower():
            detailed_error = f"Authentication failed: Your Hugging Face token may have expired or be invalid. Please reconnect. Original error: {error_msg}"
        elif "404" in error_msg:
            detailed_error = f"Resource not found: {error_msg}"
        elif "No model files found" in error_msg:
            detailed_error = f"Training data not found: {error_msg}. Please ensure your fine-tuning job completed successfully."
        else:
            detailed_error = f"Upload failed: {error_msg}"
        
        raise HTTPException(status_code=500, detail=detailed_error)

@router.post("/inference")
async def hf_inference(request: HFInferenceRequest, fastapi_request: Request):
    """Run inference using Hugging Face Inference API."""
    
    # Get OAuth info using official library
    oauth_data = get_oauth_info(fastapi_request)
    if not oauth_data:
        raise HTTPException(status_code=401, detail="Please log in with Hugging Face first")
    
    hf_token = oauth_data["access_token"]
    
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
    
    # Use official OAuth library to get user info
    oauth_data = get_oauth_info(request)
    
    if not oauth_data:
        return {
            "connected": False,
            "username": None,
            "user_info": None
        }
    
    user_info = oauth_data["user_info"]
    
    return {
        "connected": True,
        "username": user_info.get("name", ""),
        "user_info": {
            "name": user_info.get("name", ""),
            "email": user_info.get("email", ""),
            "picture": user_info.get("picture", ""),
            "is_pro": user_info.get("isPro", False)
        },
        "expires_at": oauth_data["expires_at"].isoformat()
    }

@router.get("/token-permissions")
async def check_token_permissions(request: Request):
    """Check what permissions the current HF token has."""
    
    # Get OAuth info using official library
    oauth_data = get_oauth_info(request)
    if not oauth_data:
        raise HTTPException(status_code=401, detail="Please log in with Hugging Face first")
    
    hf_token = oauth_data["access_token"]
    
    # Check rate limit before making API call
    if not check_rate_limit("permissions_check"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait a moment and try again.")
    
    try:
        # Initialize HF API
        api = HfApi(token=hf_token)
        
        # Get user info including permissions with timeout
        user_info = api.whoami()
        
        # Extract relevant permission information
        auth_info = user_info.get("auth", {})
        access_token_info = auth_info.get("accessToken", {})
        
        return {
            "user": user_info.get("name", "Unknown"),
            "email": user_info.get("email", "Unknown"),
            "permissions": {
                "role": access_token_info.get("role", "unknown"),
                "scopes": access_token_info.get("scopes", []),
            },
            "can_create_repos": "write" in str(access_token_info.get("role", "")).lower() or 
                               "write-repos" in access_token_info.get("scopes", []) or
                               "manage-repos" in access_token_info.get("scopes", []),
            "raw_auth_info": auth_info  # For debugging
        }
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate limit" in error_msg.lower():
            raise HTTPException(status_code=429, detail="HuggingFace rate limit reached. Please wait a few minutes and try again.")
        raise HTTPException(status_code=500, detail=f"Failed to check permissions: {error_msg}")
