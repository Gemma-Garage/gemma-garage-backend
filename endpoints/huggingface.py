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
from huggingface_hub import HfApi, upload_file, upload_folder, create_repo
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

# In-memory storage for OAuth states and tokens (in production, use Redis or database)
oauth_states = {}
user_tokens = {}

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

@router.get("/login")
async def huggingface_login(request: Request, request_id: str = None):
    """Initiate Hugging Face OAuth login."""
    
    if not HUGGINGFACE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Hugging Face OAuth not configured")
    
    # Generate state parameter for security
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {
        "timestamp": datetime.now(),
        "request_id": request_id  # Store request_id to use after callback
    }
    
    # Build OAuth URL
    params = {
        "client_id": HUGGINGFACE_CLIENT_ID,
        "redirect_uri": HUGGINGFACE_REDIRECT_URI,
        "scope": "read-repos write-repos manage-repos inference-api",
        "state": state,
        "response_type": "code"
    }
    
    oauth_url = f"https://huggingface.co/oauth/authorize?{urllib.parse.urlencode(params)}"
    
    # Redirect directly to Hugging Face OAuth
    return RedirectResponse(url=oauth_url)

@router.get("/callback")
async def huggingface_callback(code: str, state: str, request: Request, response: Response):
    """Handle Hugging Face OAuth callback."""
    
    # Verify state parameter
    if state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    
    # Get request_id from stored state
    stored_state = oauth_states[state]
    request_id = stored_state.get("request_id")
    
    # Clean up old states (older than 10 minutes)
    current_time = datetime.now()
    oauth_states.clear()  # Simple cleanup for demo
    
    try:
        # Exchange code for access token - single request only
        async with httpx.AsyncClient() as client:
            print("Attempting token exchange with HuggingFace...")
            token_response = await make_hf_request_with_retry(
                client,
                "POST",
                "https://huggingface.co/oauth/token",
                data={
                    "client_id": HUGGINGFACE_CLIENT_ID,
                    "client_secret": HUGGINGFACE_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": HUGGINGFACE_REDIRECT_URI
                },
                headers={"Accept": "application/json"},
                timeout=30.0
            )
        
        print(f"Token exchange response status: {token_response.status_code}")
        
        if token_response.status_code == 429:
            print("HuggingFace rate limit reached during token exchange")
            error_url = f"{FRONTEND_URL}/huggingface-test?error={urllib.parse.quote('HuggingFace is rate limiting requests. Please wait a few minutes and try again.')}"
            return RedirectResponse(url=error_url)
        
        if token_response.status_code != 200:
            error_detail = f"Failed to exchange code for token: {token_response.status_code}"
            print(f"OAuth error: {error_detail} - Response: {token_response.text[:500]}")  # Truncate long responses
            
            # Provide user-friendly error messages
            if token_response.status_code == 400:
                user_error = "Invalid authorization code. Please try logging in again."
            elif token_response.status_code == 429:
                user_error = "Too many requests. Please wait a few minutes and try again."
            else:
                user_error = f"Authentication failed ({token_response.status_code}). Please try again."
            
            error_url = f"{FRONTEND_URL}/huggingface-test?error={urllib.parse.quote(user_error)}"
            return RedirectResponse(url=error_url)
        
        token_data = token_response.json()
        access_token = token_data.get("access_token")
        
        if not access_token:
            raise HTTPException(status_code=400, detail="No access token received")
        
        # Get user info from HuggingFace API
        try:
            async with httpx.AsyncClient() as client:
                user_response = await make_hf_request_with_retry(
                    client,
                    "GET",
                    "https://huggingface.co/api/whoami",
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10.0
                )
            
            if user_response.status_code == 200:
                user_info = user_response.json()
                # Sanitize the username for repository naming
                original_name = user_info.get('name', 'Unknown')
                user_info['name'] = sanitize_username(original_name)
                print(f"Retrieved user info: {original_name} -> sanitized to: {user_info['name']}")
            else:
                # Fallback to placeholder if user info retrieval fails
                user_info = {
                    "name": "huggingface-user",  # Use valid username format
                    "email": "user@huggingface.co",
                    "note": "User info retrieval failed, using fallback"
                }
                print(f"Failed to get user info: {user_response.status_code}")
        except Exception as user_error:
            print(f"Error getting user info: {user_error}")
            # Fallback to valid username format
            user_info = {
                "name": "huggingface-user",  # Use valid username format
                "email": "user@huggingface.co",
                "note": "User info retrieval failed, using fallback"
            }
        
        # Generate session ID
        session_id = secrets.token_urlsafe(32)
        
        # Store token and user info
        user_tokens[session_id] = {
            "access_token": access_token,
            "user_info": user_info,
            "expires_at": current_time + timedelta(hours=1),  # 1 hour expiration
            "timestamp": current_time,
            "user_info_loaded": False  # Flag to indicate we need to load user info later
        }
        
        print(f"OAuth callback: Stored session {session_id} for user {user_info.get('name', 'Unknown')}")
        print(f"Total stored sessions: {len(user_tokens)}")
        
        # Create redirect response first
        if request_id:
            # Include session token in URL for cross-origin scenarios
            redirect_url = f"{FRONTEND_URL}/project/{request_id}?hf_connected=true&session_token={session_id}"
        else:
            redirect_url = f"{FRONTEND_URL}/huggingface-test?success=true&session_token={session_id}"
            
        # For debugging: let's try setting the cookie differently
        redirect_response = RedirectResponse(url=redirect_url)
        
        # Set session cookie on the redirect response
        # Use domain and path settings appropriate for cross-origin
        redirect_response.set_cookie(
            key="hf_session",
            value=session_id,
            max_age=3600,  # 1 hour
            httponly=False,  # Allow JS access for debugging
            secure=True,    # Set to True for production with HTTPS (required for cross-origin)
            samesite="none",  # Allow cross-site usage (required for cross-origin)
            domain=None,    # Don't set domain to allow cross-origin
            path="/"        # Set root path
        )
        
        print(f"OAuth callback: Redirecting to {redirect_url} with session cookie {session_id}")
        return redirect_response
        
    except Exception as e:
        print(f"OAuth callback exception: {str(e)}")
        # Redirect to frontend with error
        error_message = "Authentication failed. Please try again."
        if "rate" in str(e).lower() or "429" in str(e):
            error_message = "Too many requests. Please wait a few minutes and try again."
        
        error_url = f"{FRONTEND_URL}/huggingface-test?error={urllib.parse.quote(error_message)}"
        return RedirectResponse(url=error_url)

@router.post("/logout")
async def huggingface_logout(request: Request, response: Response):
    """Log out from Hugging Face."""
    
    # Try to get session ID from cookie or Authorization header
    session_id = request.cookies.get("hf_session")
    if not session_id:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_id = auth_header[7:]  # Remove "Bearer " prefix
    
    if session_id and session_id in user_tokens:
        del user_tokens[session_id]
        print(f"Logout: Removed session {session_id}")
    
    response.delete_cookie("hf_session")
    
    return {"success": True, "message": "Logged out successfully"}

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

def get_user_token(request: Request):
    """Get user token from session cookie or Authorization header."""
    # Try cookie first
    session_id = request.cookies.get("hf_session")
    
    # If no cookie, try Authorization header
    if not session_id:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_id = auth_header[7:]  # Remove "Bearer " prefix
    
    if not session_id or session_id not in user_tokens:
        return None
    
    token_data = user_tokens[session_id]
    
    # Check if token is expired
    if datetime.now() > token_data["expires_at"]:
        del user_tokens[session_id]
        return None
    
    return token_data

@router.post("/upload_model")
async def upload_model_to_hf(request: HFUploadRequest, fastapi_request: Request):
    """Upload a fine-tuned model to Hugging Face."""
    
    # Get user token
    token_data = get_user_token(fastapi_request)
    if not token_data:
        raise HTTPException(status_code=401, detail="Please log in with Hugging Face first")
    
    hf_token = token_data["access_token"]
    hf_username = sanitize_username(token_data["user_info"]["name"])
    
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
    
    # Get user token
    token_data = get_user_token(fastapi_request)
    if not token_data:
        raise HTTPException(status_code=401, detail="Please log in with Hugging Face first")
    
    hf_token = token_data["access_token"]
    
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
    
    # Try cookie first
    session_id = request.cookies.get("hf_session")
    source = "cookie"
    
    # If no cookie, try Authorization header
    if not session_id:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_id = auth_header[7:]  # Remove "Bearer " prefix
            source = "header"
    
    print(f"Status check: Session ID from {source}: {session_id}")
    print(f"Available sessions: {list(user_tokens.keys())}")
    
    token_data = get_user_token(request)
    
    if not token_data:
        print("Status check: No valid token data found")
        return {
            "connected": False,
            "username": None,
            "user_info": None,
            "debug": {
                "session_id": session_id,
                "source": source,
                "available_sessions": list(user_tokens.keys())
            }
        }
    
    user_info = token_data["user_info"]
    print(f"Status check: Returning connected status for user {user_info.get('name', 'Unknown')}")
    
    return {
        "connected": True,
        "username": user_info.get("name", ""),
        "user_info": {
            "name": user_info.get("name", ""),
            "email": user_info.get("email", ""),
            "picture": user_info.get("picture", ""),
            "is_pro": user_info.get("isPro", False)
        },
        "expires_at": token_data["expires_at"].isoformat(),
        "debug": {
            "session_id": session_id,
            "source": source,
            "available_sessions": list(user_tokens.keys())
        }
    }

@router.get("/token-permissions")
async def check_token_permissions(request: Request):
    """Check what permissions the current HF token has."""
    
    # Get user token
    token_data = get_user_token(request)
    if not token_data:
        raise HTTPException(status_code=401, detail="Please log in with Hugging Face first")
    
    hf_token = token_data["access_token"]
    
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
