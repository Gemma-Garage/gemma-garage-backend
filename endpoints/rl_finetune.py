from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from finetuning.vertexai import get_logs, submit_finetuning_job, run_vertexai_job
import uuid
from datetime import datetime, timezone, timedelta
from google.cloud import logging as cloud_logging
import json
import re


router = APIRouter()

class RLFinetuneJobRequest(BaseModel):
    model_name: str
    dataset_path: str # This should be the name of the file in the GCS bucket
    epochs: int
    learning_rate: float
    lora_rank: int = 4
    dataset_choice: str = "augmented"  # 'original' or 'augmented'
    qa_pairs_nbr: int | None = None   # Optional, for augmentation size
    custom_rubric: str = ""  # Custom rubric for reward evaluation

class RLTrainResponse(BaseModel):
    message: str
    request_id: str

class RLLogsResponse(BaseModel):
    request_id: str
    loss_values: list | None # Expecting a list of {"timestamp": "...", "loss": ..., "reward": ...}
    latest_timestamp: str | None # To help client with next 'since' value

@router.post("/train", response_model=RLTrainResponse)
async def train_rl_model(request: RLFinetuneJobRequest):
    request_id = str(uuid.uuid4())
    try:
        print(f"Received RL training request: {request.model_dump()}, assigning request_id: {request_id}")
        # Emit a cloud log for job submission

        cloud_logger_client = cloud_logging.Client()
        log_name = f"gemma_garage_job_logs_{request_id}"
        cloud_logger = cloud_logger_client.logger(log_name)
        cloud_logger.log_struct({
            "status_message": "RL Training job submitted and pending execution...",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": 0,
            "step_name": "RL Job Submitted"
        }, severity="INFO")

        # Determine which dataset to use based on dataset_choice
        dataset_path = request.dataset_path
        submit_finetuning_job(
            model_name=request.model_name,
            dataset_path=dataset_path, # Should be just the filename
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            lora_rank=request.lora_rank,
            request_id=request_id,
            custom_rubric=request.custom_rubric,
            job_type="rl_finetuning"
        )
        return RLTrainResponse(message="RL Training job submitted successfully.", request_id=request_id)
    except Exception as e:
        print(f"Error submitting RL training job for request_id {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit RL training job: {str(e)}")

@router.get("/logs/{request_id}", response_model=RLLogsResponse)
async def get_rl_training_logs(request_id: str, since: str | None = Query(None)):
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
            # Default to None to fetch all logs if 'since' is not provided
            since_timestamp = None
            print(f"No 'since' provided for request_id {request_id}, fetching all logs")

        if since_timestamp:
            print(f"Fetching RL logs for request_id: {request_id}, since: {since_timestamp.isoformat()}")
        else:
            print(f"Fetching all RL logs for request_id: {request_id}")
        
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
        
        return RLLogsResponse(
            request_id=request_id,
            loss_values=parsed_loss_values,
            latest_timestamp=latest_timestamp_str
        )

    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        print(f"Error fetching RL logs for request_id {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch RL logs: {str(e)}") 