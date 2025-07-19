#!/usr/bin/env python3
"""
Debug script to test token exchange response format
"""
import os
import json

# Simulate the token exchange response based on the logs
# From the logs: "Token exchange response status: 200"
# And we see: "Attempting to get user info with access token: hf_oauth_e..."

# This is what a typical HuggingFace OAuth token response looks like
sample_token_response = {
    "access_token": "hf_oauth_example_token_here",
    "token_type": "Bearer",
    "expires_in": 3600,
    "scope": "read-repos write-repos manage-repos inference-api",
    "refresh_token": "hf_refresh_example_token_here"
}

print("=== Expected Token Exchange Response ===")
print(json.dumps(sample_token_response, indent=2))

print("\n=== Token Analysis ===")
print(f"Token starts with: {sample_token_response['access_token'][:10]}...")
print(f"Token type: {sample_token_response['token_type']}")
print(f"Scopes: {sample_token_response['scope']}")

print("\n=== Authorization Header Formats ===")
print(f"Standard format: Bearer {sample_token_response['access_token']}")
print(f"Alternative format: {sample_token_response['access_token']}")

print("\n=== HuggingFace API Endpoints ===")
print("User info endpoint: https://huggingface.co/api/whoami")
print("Expected headers: Authorization: Bearer <token>")

print("\n=== Debugging Steps ===")
print("1. Check if token has correct scopes")
print("2. Verify token format in Authorization header")
print("3. Check if token is expired")
print("4. Verify HuggingFace API endpoint is correct") 