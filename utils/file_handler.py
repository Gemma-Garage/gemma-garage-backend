import os
# import shutil # No longer needed for GCS upload
from fastapi import UploadFile
import uuid
from google.cloud import storage # Added for GCS

async def save_uploaded_file(uploaded_file: UploadFile, dest_bucket_name: str) -> str:
    """
    Uploads a file to Google Cloud Storage and returns its GCS URI.
    """
    if not uploaded_file.filename:
        raise ValueError("File name cannot be empty")

    storage_client = storage.Client()
    bucket = storage_client.bucket(dest_bucket_name.replace("gs://", "")) # Remove gs:// prefix for bucket name

    blob_name = uploaded_file.filename 
    blob = bucket.blob(blob_name)

    try:
        # Upload the file
        # uploaded_file.file is a SpooledTemporaryFile, which can be read directly
        blob.upload_from_file(uploaded_file.file)
    except Exception as e:
        # Handle exceptions during upload (e.g., permissions, network issues)
        raise RuntimeError(f"Failed to upload file to GCS: {e}") from e

    # Return the GCS URI of the uploaded file
    gcs_uri = f"gs://{bucket.name}/{blob.name}"
    return gcs_uri
