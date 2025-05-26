from google.cloud import aiplatform
from google.cloud import logging
import ast
from datetime import datetime, timezone # Ensure timezone is imported
import json
import os
import time

# Get bucket names from environment variables
NEW_DATA_BUCKET = os.environ.get("NEW_DATA_BUCKET", "gs://your-default-data-bucket")
NEW_MODEL_OUTPUT_BUCKET = os.environ.get("NEW_MODEL_OUTPUT_BUCKET", "gs://your-default-model-output-bucket")
NEW_STAGING_BUCKET = os.environ.get("NEW_STAGING_BUCKET", "gs://your-default-staging-bucket")
VERTEX_AI_PROJECT = os.environ.get("VERTEX_AI_PROJECT", "llm-garage")
VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
VERTEX_AI_SERVICE_ACCOUNT = os.environ.get("VERTEX_AI_SERVICE_ACCOUNT", "513913820596-compute@developer.gserviceaccount.com")


def run_vertexai_job(model_name, dataset_path, epochs, learning_rate, lora_rank, request_id: str):
    # if not all([NEW_DATA_BUCKET, NEW_MODEL_OUTPUT_BUCKET, NEW_STAGING_BUCKET]) or \\
    #    "your-default" in NEW_DATA_BUCKET:
    #     print("WARNING: One or more GCS bucket environment variables are not set or are using default placeholder values.")

    aiplatform.init(project=VERTEX_AI_PROJECT,
                    location=VERTEX_AI_LOCATION,
                    staging_bucket=NEW_STAGING_BUCKET)

    job_display_name = f"gemma-peft-finetune-job-{request_id[:8]}" # Include part of request_id for easier identification

    # Define labels for the Vertex AI job
    job_labels = {"gemma_garage_req_id": request_id}

    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_display_name,
        container_uri="gcr.io/llm-garage/gemma-finetune:latest",
        model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest", # Optional
    )

    # Ensure dataset_path is just the filename, bucket is prepended
    # if dataset_path.startswith("gs://"):
    #     # If full path is provided, extract filename. This is a safeguard.
    #     dataset_path = dataset_path.split("/")[-1]
    #     print(f"Warning: dataset_path included gs:// prefix. Using filename: {dataset_path}")


    training_args = [
        f"--dataset={NEW_DATA_BUCKET}/{dataset_path}",
        f"--output_dir={NEW_MODEL_OUTPUT_BUCKET}/model/{request_id}", # Unique output dir per request
        f"--model_name={model_name}",
        f"--epochs={epochs}",
        f"--learning_rate={learning_rate}",
        f"--lora_rank={lora_rank}",
        f"--request_id={request_id}", # Pass request_id to the training container
        f"--project_id={VERTEX_AI_PROJECT}"  # Pass project_id for logging
    ]

    print(f"Submitting training job: {job_display_name} with request_id: {request_id} for project {VERTEX_AI_PROJECT}")
    job.run(
        base_output_dir=f"{NEW_MODEL_OUTPUT_BUCKET}/vertex_outputs/{request_id}", # Vertex AI specific outputs, unique per request
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        args=training_args,
        replica_count=1,
        sync=False, # Run asynchronously
        service_account=VERTEX_AI_SERVICE_ACCOUNT,
    )
    time.sleep(10)
    print(f"Vertex AI Job {job.display_name} (resource: {job.resource_name}) submitted with labels: {job_labels}")
    # No need to return job.resource_name if get_logs relies solely on the label for filtering.

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
def extract_loss_from_logs(logs):
    loss_values = []

    for log_entry in logs:
        payload = log_entry.get("payload") # Changed to "payload"
        timestamp = log_entry.get("timestamp")

        if not payload: # Skip if no payload
            continue

        parsed_payload_dict = None
        if isinstance(payload, dict):
            parsed_payload_dict = payload # It's already the structured log dict
        elif isinstance(payload, str):
            # If the payload is a string, attempt to parse it if it looks like a dict string
            # This handles cases where a stringified dict might have been logged.
            if payload.strip().startswith("{") and payload.strip().endswith("}"):
                try:
                    parsed_payload_dict = ast.literal_eval(payload)
                except (ValueError, SyntaxError) as e:
                    # print(f"Could not parse string payload as dict: {payload}, Error: {e}") # Debug
                    pass # Continue, as it might be a plain string with "loss" later
            # If it's a plain string, we might still search for "loss" if needed,
            # but primary expectation is structured logs or stringified dicts from them.
            # For now, if it's a plain string and not a dict-string, we won't find 'loss' via key access.
            # A simple "loss" in text check could be added here for very basic string logs if necessary.
            # if not parsed_payload_dict and "loss" in payload.lower():
            #    print(f"Found 'loss' in plain string, but no structured data: {payload}")


        if isinstance(parsed_payload_dict, dict) and "loss" in parsed_payload_dict:
            loss = parsed_payload_dict.get("loss") # Use .get for safety
            # Also, other fields like 'epoch', 'step', 'learning_rate' are in parsed_payload_dict
            if isinstance(loss, (int, float)): # Ensure loss is a number
                loss_values.append({ # Store as a dictionary directly
                    "timestamp": timestamp, 
                    "loss": loss,
                    # Optionally include other metrics if needed by frontend later
                    # "epoch": parsed_payload_dict.get("epoch"),
                    # "step": parsed_payload_dict.get("step"),
                    # "learning_rate": parsed_payload_dict.get("learning_rate")
                })
            else:
                print(f"Warning: Parsed loss is not a number: {loss} from payload: {parsed_payload_dict}")
        # else:
            # print(f"Debug: No 'loss' key in parsed_payload_dict or payload not a dict: {parsed_payload_dict}")


    if not loss_values:
        # print(f"No loss values extracted from {len(logs)} log entries.") # Debug
        return json.dumps([]) # Return empty JSON array string if no loss values

    # Sort by timestamp before returning
    # The items in loss_values are now dicts: {"timestamp": ..., "loss": ...}
    loss_values.sort(key=lambda x: datetime.fromisoformat(x["timestamp"].replace("Z", "+00:00")))

    # print(f"Extracted {len(loss_values)} loss entries from logs.")
    # print(loss_values) # This will now be a list of dicts
    
    # The endpoint expects a JSON string which it will then parse.
    # The list of dicts is already what the frontend expects for data.loss_values after JSON parsing.
    return json.dumps(loss_values, indent=2)

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