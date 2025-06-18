from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from google.cloud import storage
from datetime import timedelta
import os
import io
import zipfile

router = APIRouter()

def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parses a GCS path (gs://bucket/prefix/object) into bucket_name and object_prefix."""
    if not gcs_path.startswith("gs://"):
        raise ValueError("GCS path must start with gs://")
    path_without_scheme = gcs_path[5:]
    if "/" not in path_without_scheme:
        bucket_name = path_without_scheme
        object_prefix = "" # No prefix, means root of the bucket or just bucket itself
    else:
        parts = path_without_scheme.split("/", 1)
        bucket_name = parts[0]
        object_prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, object_prefix

@router.post("/download_weights")
async def generate_gcs_download_urls(request: Request): # Renamed for clarity
    body = await request.json()
    request_id = body.get("request_id")

    if not request_id:
        raise HTTPException(status_code=400, detail="request_id not provided")

    base_gcs_output_bucket_uri = os.environ.get("NEW_MODEL_OUTPUT_BUCKET")
    if not base_gcs_output_bucket_uri:
        raise HTTPException(status_code=500, detail="NEW_MODEL_OUTPUT_BUCKET environment variable not set")

    # Corrected path construction to point to where model artifacts are saved by the training script
    # Training script saves to: {NEW_MODEL_OUTPUT_BUCKET}/model/{request_id}
    # Example: gs://my-bucket-name/actual-path-prefix/model/request_id/
    gcs_model_directory_path = f"{base_gcs_output_bucket_uri.rstrip('/')}/model/{request_id}/"

    try:
        bucket_name, model_directory_prefix = parse_gcs_path(gcs_model_directory_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid GCS path format constructed: {gcs_model_directory_path} - {str(e)}")

    storage_client = storage.Client()
    
    try:
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        print(f"Error initializing storage client or getting bucket: {e}")
        raise HTTPException(status_code=500, detail=f"Error accessing GCS bucket '{bucket_name}': {str(e)}")

    blobs = list(bucket.list_blobs(prefix=model_directory_prefix))

    if not blobs:
        raise HTTPException(
            status_code=404, 
            detail=f"No files found in GCS path {gcs_model_directory_path}. Ensure training completed and saved files."
        )

    files_to_download = []
    try:
        for blob in blobs:
            # Skip if it's a "directory" placeholder object (some GCS operations create these)
            if blob.name.endswith('/'):
                continue
            
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=1),
                method="GET",
            )
            # Get just the filename from the full blob path
            file_name = blob.name.split('/')[-1]
            files_to_download.append({"filename": file_name, "download_url": signed_url})

        if not files_to_download:
             raise HTTPException(
                status_code=404, 
                detail=f"No downloadable files (non-folders) found in GCS path {gcs_model_directory_path}."
            )

        return JSONResponse({"files": files_to_download})

    except Exception as e:
        print(f"Error generating signed URLs for files in {gcs_model_directory_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not generate download URLs: {str(e)}")

@router.post("/download_weights_zip")
async def download_weights_zip(request: Request):
    body = await request.json()
    request_id = body.get("request_id")

    if not request_id:
        raise HTTPException(status_code=400, detail="request_id not provided")

    # The model output path is always: {NEW_MODEL_OUTPUT_BUCKET}/model/{request_id}/final_model
    base_gcs_output_bucket_uri = os.environ.get("NEW_MODEL_OUTPUT_BUCKET")
    if not base_gcs_output_bucket_uri:
        raise HTTPException(status_code=500, detail="NEW_MODEL_OUTPUT_BUCKET environment variable not set")

    gcs_model_directory_path = f"{base_gcs_output_bucket_uri.rstrip('/')}/model/{request_id}/final_model/"

    try:
        bucket_name, model_directory_prefix = parse_gcs_path(gcs_model_directory_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid GCS path format constructed: {gcs_model_directory_path} - {str(e)}")

    storage_client = storage.Client()
    try:
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        print(f"Error initializing storage client or getting bucket: {e}")
        raise HTTPException(status_code=500, detail=f"Error accessing GCS bucket '{bucket_name}': {str(e)}")

    blobs = list(bucket.list_blobs(prefix=model_directory_prefix))
    if not blobs:
        raise HTTPException(
            status_code=404, 
            detail=f"No files found in GCS path {gcs_model_directory_path}. Ensure training completed and saved files."
        )

    # Create a zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for blob in blobs:
            if blob.name.endswith('/'):
                continue
            file_data = blob.download_as_bytes()
            arcname = blob.name.split('/')[-1]
            zipf.writestr(arcname, file_data)
    zip_buffer.seek(0)

    return FileResponse(zip_buffer, media_type="application/zip", filename=f"model_{request_id}.zip")
