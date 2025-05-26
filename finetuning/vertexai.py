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
        # Ensure payload is a string, sometimes it can be a dict
        payload_text = None
        if isinstance(entry.payload, str):
            payload_text = entry.payload
        elif isinstance(entry.payload, dict) and 'message' in entry.payload: # Common for structured logs
            payload_text = entry.payload['message']
        
        if payload_text: # Only append if we have text
            logs.append({
                "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                "textPayload": payload_text
            })

    print(f"Retrieved {len(logs)} log entries for request_id {request_id}.")
    # print(logs) # Can be very verbose

    return extract_loss_from_logs(logs) # extract_loss_from_logs sorts by timestamp

# 🧠 Function to parse logs and extract loss values
def extract_loss_from_logs(logs):
    loss_values = []

    for log_entry in logs: # Renamed log to log_entry to avoid conflict if 'log' is a var name
        text = log_entry.get("textPayload")
        timestamp = log_entry.get("timestamp")

        if text and "loss" in text: # Simple check, might need refinement
            try:
                # Attempt to parse if it's a dict string: "{'loss': 0.123, ...}"
                # More robust parsing might be needed depending on actual log format
                if isinstance(text, str) and text.strip().startswith("{") and text.strip().endswith("}"):
                    parsed = ast.literal_eval(text)
                elif isinstance(text, dict): # If already a dict (e.g. from structured logging)
                    parsed = text
                else: # If it's just a string that contains "loss", try to find it
                    # This part is tricky and depends on log format.
                    # For now, we rely on ast.literal_eval or direct dict access.
                    # A regex might be needed for plain string logs: e.g., re.search(r"loss:\s*([0-9.]+)", text)
                    parsed = None 

                if isinstance(parsed, dict) and "loss" in parsed:
                    loss = parsed["loss"]
                    if isinstance(loss, (int, float)): # Ensure loss is a number
                        loss_values.append((timestamp, loss))
                    else:
                        print(f"Warning: Parsed loss is not a number: {loss} from log: {text}")

            except (ValueError, SyntaxError) as e:
                # print(f"Could not parse text for loss: {text}, Error: {e}") # Can be noisy
                continue # Skip if parsing fails

    if not loss_values:
        return None # Return None if no loss values found

    # Sort by timestamp (original get_logs used DESCENDING, then sorted. Now using ASCENDING, so sorting here is still important)
    loss_values.sort(key=lambda x: datetime.fromisoformat(x[0].replace("Z", "+00:00")))

    # print(f"Extracted {len(loss_values)} loss entries from logs.")
    # print(loss_values)
    return json.dumps([{"timestamp": t, "loss": l} for t, l in loss_values], indent=2)

# Remove or comment out the __main__ block if it's not needed for direct script execution
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