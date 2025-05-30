from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from google.cloud import storage
from datetime import timedelta
import os

router = APIRouter()

def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parses a GCS path (gs://bucket/prefix/object) into bucket_name and object_prefix."""
    if not gcs_path.startswith("gs://"):
        raise ValueError("GCS path must start with gs://")
    path_without_scheme = gcs_path[5:]
    if "/" not in path_without_scheme:
        # Path is likely just "gs://bucketname" which means prefix is empty
        bucket_name = path_without_scheme
        object_prefix = ""
    else:
        parts = path_without_scheme.split("/", 1)
        bucket_name = parts[0]
        object_prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, object_prefix

@router.post("/download_weights")
async def generate_gcs_download_url(request: Request):
    body = await request.json()
    request_id = body.get("request_id")

    if not request_id:
        raise HTTPException(status_code=400, detail="request_id not provided")

    # Use the existing NEW_MODEL_OUTPUT_BUCKET environment variable
    base_gcs_output_bucket = os.environ.get("NEW_MODEL_OUTPUT_BUCKET")
    if not base_gcs_output_bucket:
        raise HTTPException(status_code=500, detail="NEW_MODEL_OUTPUT_BUCKET environment variable not set")

    # Construct the GCS directory path using the base bucket and request_id
    # This should match the output structure: gs://<NEW_MODEL_OUTPUT_BUCKET>/vertex_outputs/<request_id>/
    gcs_directory_path = f"{base_gcs_output_bucket.rstrip('/')}/vertex_outputs/{request_id}/"

    try:
        bucket_name, directory_prefix = parse_gcs_path(gcs_directory_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid GCS path format constructed: {gcs_directory_path} - {str(e)}")

    # Ensure directory_prefix for a directory ends with a '/' (already handled by f-string above, but good for robustness)
    if directory_prefix and not directory_prefix.endswith('/'):
        directory_prefix += '/'
    
    storage_client = storage.Client()
    
    try:
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        # Handle cases where bucket might not be accessible or client init fails
        print(f"Error initializing storage client or getting bucket: {e}") # Basic logging
        raise HTTPException(status_code=500, detail=f"Error accessing GCS bucket \'{bucket_name}\': {str(e)}")

    target_blob_obj = None
    found_filename = None
    # These are the base filenames we expect in the PEFT adapter output directory
    preferred_filenames = ["adapter_model.safetensors", "adapter_model.bin"] 
    
    for filename in preferred_filenames:
        # Construct the full blob name (object path within the bucket)
        blob_name_to_check = directory_prefix + filename
        
        blob = bucket.get_blob(blob_name_to_check)
        if blob and blob.exists(): # Check if blob exists
            target_blob_obj = blob
            found_filename = filename # This is the base filename, e.g., "adapter_model.safetensors"
            break # Found the first preferred file, stop searching
            
    if not target_blob_obj:
        raise HTTPException(
            status_code=404, 
            detail=f"Could not find \'adapter_model.safetensors\' or \'adapter_model.bin\' in GCS path {gcs_directory_path}"
        )

    try:
        # Generate a signed URL for the blob, expiring in 1 hour
        signed_url = target_blob_obj.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=1),
            method="GET",
        )
        # found_filename is already the base name (e.g., "adapter_model.safetensors")
        # which is suitable for the download attribute in HTML <a> tag.
        return JSONResponse({"download_url": signed_url, "filename": found_filename})

    except Exception as e:
        print(f"Error generating signed URL for {target_blob_obj.name}: {e}") # Basic logging
        raise HTTPException(status_code=500, detail=f"Could not generate download URL: {str(e)}")

# The old GET endpoint /download_file is removed as it's no longer needed.
# The frontend will use the signed_url directly to download from GCS.
