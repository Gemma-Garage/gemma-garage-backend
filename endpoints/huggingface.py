import os
import asyncio
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import httpx
from google.cloud import firestore
from typing import Optional
import json
import secrets
import urllib.parse
from datetime import datetime, timedelta
from huggingface_hub import HfApi, upload_file, create_repo
from config_vars import NEW_DATA_BUCKET

router = APIRouter()

# Initialize Firestore client
db = firestore.Client()

# OAuth configuration
HUGGINGFACE_CLIENT_ID = os.getenv("HUGGINGFACE_CLIENT_ID")
HUGGINGFACE_CLIENT_SECRET = os.getenv("HUGGINGFACE_CLIENT_SECRET")
HUGGINGFACE_REDIRECT_URI = os.getenv("HUGGINGFACE_REDIRECT_URI", "http://localhost:8080/oauth/huggingface/callback")

# In-memory storage for OAuth states and tokens (in production, use Redis or database)
oauth_states = {}
user_tokens = {}

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

@router.get("/config")
async def check_oauth_config():
    """Check OAuth configuration status."""
    return {
        "client_id_configured": bool(HUGGINGFACE_CLIENT_ID),
        "client_secret_configured": bool(HUGGINGFACE_CLIENT_SECRET),
        "redirect_uri": HUGGINGFACE_REDIRECT_URI,
        "client_id_preview": HUGGINGFACE_CLIENT_ID[:8] + "..." if HUGGINGFACE_CLIENT_ID else None
    }

@router.get("/login")
async def huggingface_login(request: Request):
    """Initiate Hugging Face OAuth login."""
    
    if not HUGGINGFACE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Hugging Face OAuth not configured")
    
    # Generate state parameter for security
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {"timestamp": datetime.now()}
    
    # Build OAuth URL
    params = {
        "client_id": HUGGINGFACE_CLIENT_ID,
        "redirect_uri": HUGGINGFACE_REDIRECT_URI,
        "scope": "read-repos write-repos inference-api",
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
    
    # Clean up old states (older than 10 minutes)
    current_time = datetime.now()
    oauth_states.clear()  # Simple cleanup for demo
    
    try:
        # Exchange code for access token
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://huggingface.co/oauth/token",
                data={
                    "client_id": HUGGINGFACE_CLIENT_ID,
                    "client_secret": HUGGINGFACE_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": HUGGINGFACE_REDIRECT_URI
                },
                headers={"Accept": "application/json"}
            )
        
        if token_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to exchange code for token")
        
        token_data = token_response.json()
        access_token = token_data.get("access_token")
        
        if not access_token:
            raise HTTPException(status_code=400, detail="No access token received")
        
        # Get user info
        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {access_token}"}
            )
        
        if user_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get user info")
        
        user_info = user_response.json()
        
        # Generate session ID
        session_id = secrets.token_urlsafe(32)
        
        # Store token and user info
        user_tokens[session_id] = {
            "access_token": access_token,
            "user_info": user_info,
            "expires_at": current_time + timedelta(hours=1),  # 1 hour expiration
            "timestamp": current_time
        }
        
        # Set session cookie
        response.set_cookie(
            key="hf_session",
            value=session_id,
            max_age=3600,  # 1 hour
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax"
        )
        
        # Redirect to frontend success page
        return RedirectResponse(url="http://localhost:3000/hf-test?success=true")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OAuth callback error: {str(e)}")

@router.post("/logout")
async def huggingface_logout(request: Request, response: Response):
    """Log out from Hugging Face."""
    
    session_id = request.cookies.get("hf_session")
    if session_id and session_id in user_tokens:
        del user_tokens[session_id]
    
    response.delete_cookie("hf_session")
    
    return {"success": True, "message": "Logged out successfully"}

def get_user_token(request: Request):
    """Get user token from session."""
    session_id = request.cookies.get("hf_session")
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
    hf_username = token_data["user_info"]["name"]
    
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
    
    token_data = get_user_token(request)
    
    if not token_data:
        return {
            "connected": False,
            "username": None,
            "user_info": None
        }
    
    user_info = token_data["user_info"]
    
    return {
        "connected": True,
        "username": user_info.get("name", ""),
        "user_info": {
            "name": user_info.get("name", ""),
            "email": user_info.get("email", ""),
            "picture": user_info.get("picture", ""),
            "is_pro": user_info.get("isPro", False)
        },
        "expires_at": token_data["expires_at"].isoformat()
    }
