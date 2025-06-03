from google.cloud import logging # Keep logging
import ast
from datetime import datetime, timezone # Ensure timezone is imported
import json
import os
import requests

# Get bucket names from environment variables
NEW_DATA_BUCKET = os.environ.get("NEW_DATA_BUCKET", "gs://your-default-data-bucket")
NEW_MODEL_OUTPUT_BUCKET = os.environ.get("NEW_MODEL_OUTPUT_BUCKET", "gs://your-default-model-output-bucket")
NEW_STAGING_BUCKET = os.environ.get("NEW_STAGING_BUCKET", "gs://your-default-staging-bucket")
VERTEX_AI_PROJECT = os.environ.get("VERTEX_AI_PROJECT", "llm-garage")
VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
VERTEX_AI_SERVICE_ACCOUNT = os.environ.get("VERTEX_AI_SERVICE_ACCOUNT", "513913820596-compute@developer.gserviceaccount.com")

def submit_finetuning_job(
    model_name: str,
    dataset_path: str,
    epochs: int,
    learning_rate: float,
    lora_rank: int = 4,
    request_id: str = None):

    output_dir = f"{NEW_MODEL_OUTPUT_BUCKET}/model/{request_id}"
    project_id= VERTEX_AI_PROJECT
    url = "https://llm-garage-finetune-513913820596.us-central1.run.app/run-finetune-job"
    payload = {
    "dataset": f"{NEW_DATA_BUCKET}/{dataset_path}",
    "output_dir": output_dir,
    "model_name": model_name,
    "epochs": epochs,
    "learning_rate": learning_rate,
    "lora_rank": lora_rank,
    "request_id": request_id,
    "project_id": project_id
    }
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print("Job submitted successfully:", response.json())
    else:
        print("Failed to submit job:", response.status_code, response.text)


def get_logs(
    request_id: str,
    since_timestamp: datetime,
    project_id: str = VERTEX_AI_PROJECT, # Use from env var
    limit: int = 200 
):
    client = logging.Client(project=project_id)

    if since_timestamp.tzinfo is None:
        since_timestamp = since_timestamp.replace(tzinfo=timezone.utc)

    # Define the convention for the custom log name
    # This MUST match the log name used in your gemma-garage-finetuning/src/finetuning.py script
    custom_log_name = f"gemma_garage_job_logs_{request_id}"

    # Updated filter_str to query the custom log name
    filter_str = (
        f'logName="projects/{project_id}/logs/{custom_log_name}" '
        f'AND timestamp >= "{since_timestamp.isoformat()}"'
        # Optional: You might still want to filter by severity if your custom logs have it
        # f'AND severity >= DEFAULT ' 
    )
    
    print(f"Querying logs with new filter for custom log name: {filter_str}")

    entries = client.list_entries(
        filter_=filter_str,
        order_by=logging.ASCENDING, 
        max_results=limit,
    )

    logs = []
    for entry in entries:
        payload_to_process = None
        if isinstance(entry.payload, dict): # If it's already a dict (our structured log)
            payload_to_process = entry.payload 
        elif isinstance(entry.payload, str): # If it's a simple string log
            payload_to_process = entry.payload
        
        if payload_to_process: # Only append if we have a payload
            logs.append({
                "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                "payload": payload_to_process # Changed "textPayload" to "payload"
            })

    print(f"Retrieved {len(logs)} log entries for request_id {request_id}.")
    # print(logs) # Can be very verbose

    # The response from extract_loss_from_logs should be a JSON string 
    # containing a list of {"timestamp": ..., "loss": ...} objects, or None.
    # The endpoint handler in finetune.py will then parse this string.
    return extract_loss_from_logs(logs)

# 🧠 Function to parse logs and extract loss values
def extract_loss_from_logs(logs): # logs is a list of {"timestamp": ..., "payload": ...}
    processed_logs = []
    for log_entry in logs:
        payload = log_entry.get("payload")
        timestamp = log_entry.get("timestamp")

        if not payload: # Skip if no payload or timestamp
            continue

        entry_to_add = None
        if isinstance(payload, dict):
            entry_to_add = payload.copy() # It's already the structured log dict
        elif isinstance(payload, str):
            # If the payload is a string, attempt to parse it if it looks like a dict string
            if payload.strip().startswith("{") and payload.strip().endswith("}"):
                try:
                    parsed_dict = ast.literal_eval(payload)
                    if isinstance(parsed_dict, dict):
                        entry_to_add = parsed_dict
                    else: # literal_eval returned non-dict
                        entry_to_add = {"message": payload}
                except (ValueError, SyntaxError):
                    # It's a string but not a parsable dict string, treat as simple message
                    entry_to_add = {"message": payload}
            else:
                # Plain string message
                entry_to_add = {"message": payload}
        
        if entry_to_add is not None:
            # Ensure timestamp from the log entry is consistently part of the object sent to frontend
            entry_to_add["timestamp"] = timestamp 
            processed_logs.append(entry_to_add)

    if not processed_logs:
        return json.dumps([]) # Return empty JSON array string

    # Sort by timestamp before returning
    # Handle cases where timestamp might be None in sorting key
    processed_logs.sort(
        key=lambda x: datetime.fromisoformat(x["timestamp"].replace("Z", "+00:00")) if x.get("timestamp") else datetime.min.replace(tzinfo=timezone.utc)
    )
    
    # The endpoint expects a JSON string which it will then parse.
    # This list of dictionaries is what the frontend expects for data.loss_values after JSON parsing.
    return json.dumps(processed_logs, indent=2)

# Remove or comment out the __main__ block if it\'s not needed for direct script execution
# or adapt it for testing the new functions.
# if __name__ == "__main__":
#     # Example usage (requires a valid request_id that has logs)
#     # test_request_id = "your_test_request_id_here" 
#     # test_since_timestamp = datetime.now(timezone.utc) - timedelta(hours=1)
#     # logs_json_str = get_logs(request_id=test_request_id, since_timestamp=test_since_timestamp)
#     # if logs_json_str:
#     #     print("Losses:")
#     #     print(json.dumps(json.loads(logs_json_str), indent=2))
#     # else:
#     #     print("No loss logs found.")
#     pass