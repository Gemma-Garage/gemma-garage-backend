#!/usr/bin/env python3
"""
Debug script to test OAuth token exchange with HuggingFace
"""
import os
import httpx
import asyncio

# Get environment variables
HUGGINGFACE_CLIENT_ID = os.getenv("HUGGINGFACE_CLIENT_ID")
HUGGINGFACE_CLIENT_SECRET = os.getenv("HUGGINGFACE_CLIENT_SECRET")
HUGGINGFACE_REDIRECT_URI = "https://llm-garage-513913820596.us-central1.run.app/oauth/huggingface/callback"

async def test_token_exchange():
    """Test the token exchange with a dummy code to see the exact error."""
    
    print("=== OAuth Configuration ===")
    print(f"Client ID: {HUGGINGFACE_CLIENT_ID[:8]}..." if HUGGINGFACE_CLIENT_ID else "NOT SET")
    print(f"Client Secret: {HUGGINGFACE_CLIENT_SECRET[:8]}..." if HUGGINGFACE_CLIENT_SECRET else "NOT SET")
    print(f"Redirect URI: {HUGGINGFACE_REDIRECT_URI}")
    
    # Test with a dummy code (this will fail, but we'll see the exact error)
    dummy_code = "dummy_authorization_code"
    
    print("\n=== Testing Token Exchange ===")
    print("Using dummy code to test request format...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://huggingface.co/oauth/token",
                data={
                    "client_id": HUGGINGFACE_CLIENT_ID,
                    "client_secret": HUGGINGFACE_CLIENT_SECRET,
                    "code": dummy_code,
                    "grant_type": "authorization_code",
                    "redirect_uri": HUGGINGFACE_REDIRECT_URI
                },
                headers={"Accept": "application/json"},
                timeout=30.0
            )
            
            print(f"Response Status: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            print(f"Response Body: {response.text}")
            
            if response.status_code == 400:
                print("\n✅ Request format is correct (400 is expected for dummy code)")
                print("The issue is likely with the actual authorization code or timing")
            elif response.status_code == 401:
                print("\n❌ Client credentials are invalid")
            elif response.status_code == 403:
                print("\n❌ Access denied - check app permissions")
            else:
                print(f"\n❓ Unexpected status code: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_token_exchange()) 