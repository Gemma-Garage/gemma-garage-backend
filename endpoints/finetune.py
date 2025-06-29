from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from finetuning.vertexai import get_logs, submit_finetuning_job, run_vertexai_job
import uuid
from datetime import datetime, timezone, timedelta
import json
import re


router = APIRouter()

class FinetuneJobRequest(BaseModel):
    model_name: str
    dataset_path: str # This should be the name of the file in the GCS bucket
    epochs: int
    learning_rate: float
    lora_rank: int = 4
    dataset_choice: str = "augmented"  # 'original' or 'augmented'
    qa_pairs_nbr: int | None = None   # Optional, for augmentation size

class TrainResponse(BaseModel):
    message: str
    request_id: str

class LogsResponse(BaseModel):
    request_id: str
    loss_values: list | None # Expecting a list of {"timestamp": "...", "loss": ...}
    latest_timestamp: str | None # To help client with next 'since' value

@router.post("/train", response_model=TrainResponse)
async def train_model(request: FinetuneJobRequest):
    request_id = str(uuid.uuid4())
    try:
        print(f"Received training request: {request.model_dump()}, assigning request_id: {request_id}")
        # Determine which dataset to use based on dataset_choice
        dataset_path = request.dataset_path
        if request.dataset_choice == "augmented":
            # If the user chose augmented, replace .json with _augmented.json
            if '.' in dataset_path:
                dataset_path = re.sub(r'\.[^.]+$', '_augmented.json', dataset_path)
            # If qa_pairs_nbr is provided, call the augmentation API with this parameter
            if request.qa_pairs_nbr:
                # Import and call the augmentation endpoint logic directly if needed, or trigger augmentation here
                # Example: augment_dataset_gemma(dataset_path, qa_pairs_nbr=request.qa_pairs_nbr)
                print(f"[INFO] Would trigger augmentation with qa_pairs_nbr={request.qa_pairs_nbr} for {dataset_path}")
                # If augmentation is already done in frontend, this is just for completeness
        submit_finetuning_job(
            model_name=request.model_name,
            dataset_path=dataset_path, # Should be just the filename
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            lora_rank=request.lora_rank,
            request_id=request_id
        )
        return TrainResponse(message="Training job submitted successfully.", request_id=request_id)
    except Exception as e:
        print(f"Error submitting training job for request_id {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit training job: {str(e)}")

@router.get("/logs/{request_id}", response_model=LogsResponse)
async def get_training_logs(request_id: str, since: str | None = Query(None)):
    try:
        since_timestamp: datetime
        if since:
            try:
                # Attempt to parse with 'Z' first, then without if it fails
                try:
                    since_timestamp = datetime.fromisoformat(since)
                except ValueError:
                    since_timestamp = datetime.fromisoformat(since.replace("Z", "+00:00"))
                
                if since_timestamp.tzinfo is None: # Ensure timezone aware
                    since_timestamp = since_timestamp.replace(tzinfo=timezone.utc)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid 'since' timestamp format. Use ISO 8601 (e.g., YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DDTHH:MM:SS+00:00).")
        else:
            # Default to 24 hours ago if 'since' is not provided
            since_timestamp = datetime.now(timezone.utc) - timedelta(days=1)
            print(f"No 'since' provided for request_id {request_id}, defaulting to {since_timestamp.isoformat()}")

        print(f"Fetching logs for request_id: {request_id}, since: {since_timestamp.isoformat()}")
        
        loss_values_json_str = get_logs(
            request_id=request_id,
            since_timestamp=since_timestamp
        )

        parsed_loss_values = None
        latest_timestamp_str = None

        if loss_values_json_str:
            try:
                parsed_loss_values = json.loads(loss_values_json_str)
                if isinstance(parsed_loss_values, list) and parsed_loss_values:
                    # Assuming logs are sorted by timestamp ascending by extract_loss_from_logs
                    latest_timestamp_str = parsed_loss_values[-1].get("timestamp")
            except json.JSONDecodeError:
                print(f"Failed to parse loss_values_json_str for request_id {request_id}: {loss_values_json_str}")
                # Return None for loss_values if parsing fails.
                pass 
        
        return LogsResponse(
            request_id=request_id,
            loss_values=parsed_loss_values,
            latest_timestamp=latest_timestamp_str
        )

    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        print(f"Error fetching logs for request_id {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {str(e)}")
