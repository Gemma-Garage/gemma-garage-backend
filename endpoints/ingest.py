import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
from google.cloud import storage
from urllib.parse import urlparse
from gitingest import ingest
from config_vars import NEW_DATA_BUCKET

class IngestRequest(BaseModel):
    repository_url: str

router = APIRouter()

# --- Configuration ---
DESTINATION_FOLDER = "ingested_repositories/"

def upload_to_gcs(bucket_name, destination_blob_name, data):
    """Uploads data to a GCS bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name.replace("gs://", ""))
        blob = bucket.blob(destination_blob_name)

        # Upload the JSON data
        blob.upload_from_string(
            data=json.dumps(data, indent=4),
            content_type="application/json"
        )
        gcs_path = f"gs://{bucket.name}/{destination_blob_name}"
        print(f"Successfully uploaded to {gcs_path}")
        return gcs_path
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading to GCS: {e}")


@router.post("/ingest")
async def ingest_repository(request: IngestRequest):
    """
    Ingests a GitHub repository, combines its summary, tree, and content into a single JSON file,
    and uploads it to a specified GCS bucket.
    """
    repo_url = request.repository_url
    print(f"URL {repo_url} received for ingestion.")

    try:
        loop = asyncio.get_running_loop()
        
        # Run the synchronous function `ingest` in a separate thread to avoid
        # blocking the event loop and to allow it to run its own asyncio.run()
        summary, tree, content = await loop.run_in_executor(
            None,  # Use the default thread pool executor
            ingest,  # The synchronous function to call
            repo_url  # The argument for the function
        )

        # Prepare the data for upload
        ingested_data = {
            "repository_url": repo_url,
            "summary": summary,
            "tree": tree,
            "content": content
        }

        # Create a filename from the repo URL
        parsed_url = urlparse(repo_url)
        repo_name = parsed_url.path.strip('/').replace('/', '_')
        destination_blob_name = f"{DESTINATION_FOLDER}{repo_name}.json"

        # Upload the combined data
        gcs_path = upload_to_gcs(NEW_DATA_BUCKET, destination_blob_name, ingested_data)

        return {
            "message": "Repository ingested successfully.",
            "gcs_path": gcs_path,
            "summary": summary,
            "content_preview": content[:200]  # Return first 200 characters of content
        }

    except Exception as e:
        print(f"An error occurred during ingestion for URL: {repo_url}. Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
