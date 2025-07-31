from google.cloud import aiplatform, logging
import ast
from datetime import datetime, timezone # Ensure timezone is imported
from google.cloud import batch_v1
from google.protobuf.duration_pb2 import Duration
import json
import os
import time
import uuid
import argparse
import requests
import subprocess
import google.auth
import google.auth.transport.requests


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
    request_id: str = None,
    custom_rubric: str = "",
    job_type: str = "supervised"):

    job_name = "llm-garage-finetune"
    project_id = "llm-garage"
    region = "us-central1"
    output_dir = f"{NEW_MODEL_OUTPUT_BUCKET}/model/{request_id}"

    args = {
    "dataset": f"{NEW_DATA_BUCKET}/{dataset_path}",
    "output_dir": output_dir,
    "model_name": model_name,
    "epochs": epochs,
    "learning_rate": learning_rate,
    "lora_rank": lora_rank,
    "request_id": request_id,
    "project_id": project_id,
    "job_type": job_type
    }
    
    # Add custom rubric for RL jobs
    if job_type == "rl_finetuning" and custom_rubric:
        args["custom_rubric"] = custom_rubric
    args_list = [f"--{key}={value}" for key, value in args.items() if value is not None]

    command = [
        "gcloud", "beta", "run", "jobs", "execute", job_name,
        "--project", project_id,
        "--region", region,
        "--wait",
        "--args=" + ",".join(args_list)
    ]

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(google.auth.transport.requests.Request())
    access_token = credentials.token

    url = (
        f"https://{region}-run.googleapis.com/apis/run.googleapis.com/"
        f"v1/namespaces/{project_id}/jobs/{job_name}:run"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    body = {
        "overrides": {
            "containerOverrides": [
                {
                    "args": args_list
                }
            ]
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(body))

    if response.status_code == 200:
        print("✅ Job triggered successfully.")
        print(response.json())
    else:
        print("❌ Failed to trigger job:")
        print(response.status_code, response.text)

# def submit_finetuning_job(
#     model_name: str,
#     dataset_path: str,
#     epochs: int,
#     learning_rate: float,
#     lora_rank: int = 4,
#     request_id: str = None):

#     output_dir = f"{NEW_MODEL_OUTPUT_BUCKET}/model/{request_id}"
#     project_id= VERTEX_AI_PROJECT
#     url = "https://llm-garage-finetune-513913820596.us-central1.run.app/run-finetune-job"
#     payload = {
#     "dataset": f"{NEW_DATA_BUCKET}/{dataset_path}",
#     "output_dir": output_dir,
#     "model_name": model_name,
#     "epochs": epochs,
#     "learning_rate": learning_rate,
#     "lora_rank": lora_rank,
#     "request_id": request_id,
#     "project_id": project_id
#     }
#     response = requests.post(url, json=payload)

#     if response.status_code == 200:
#         print("Job submitted successfully:", response.json())
#     else:
#         print("Failed to submit job:", response.status_code, response.text)

# def submit_finetuning_job(
#     model_name: str,
#     dataset_path: str,
#     epochs: int,
#     learning_rate: float,
#     lora_rank: int = 4,
#     request_id: str = None):

# # --- Hardcoded Infrastructure & Configuration Parameters (INSIDE FUNCTION) ---
#     gcp_project_id = "llm-garage"  # Your Google Cloud Project ID
#     gcp_region = "us-central1"     # The region to run Batch jobs in

#     # GCS Bucket Paths (ensure these buckets exist and have correct permissions)
#     # REPLACE THESE WITH YOUR ACTUAL BUCKET NAMES
#     new_model_output_bucket = "gs://your-llm-garage-output-bucket"
#     new_data_bucket = "gs://your-llm-garage-data-bucket"

#     # Docker Image URI
#     image_uri = "gcr.io/llm-garage/gemma-finetune:latest" # Using project_id in image path

#     # Machine and GPU Configuration for NVIDIA_L4
#     machine_type = "g2-standard-4"
#     cpu_milli = 3500               # Requesting ~3.5 vCPUs
#     memory_mib = 14 * 1024         # Requesting 14 GiB (14 * 1024 MiB)
#     boot_disk_mib = 100 * 1024     # 100 GiB for the boot disk
#     max_run_duration_seconds = 4 * 3600 # 4 hours job timeout

#     # Optional: Service account for the Batch VMs. If None, uses default GCE SA.
#     # batch_job_sa_email = "my-batch-vm-sa@llm-garage.iam.gserviceaccount.com" # Uncomment and set if needed
#     batch_job_sa_email = None

#     # Optional: Hugging Face token if needed at runtime by the training script
#     # hf_token_runtime = "hf_yourHuggingFaceToken" # Uncomment and set if needed
#     hf_token_runtime = None
#     # --- End of Hardcoded Parameters ---

#     if not request_id:
#         request_id = f"req-{uuid.uuid4().hex[:8]}"

#     job_timestamp_suffix = f"{int(time.time())}"
#     safe_model_name = model_name.split('/')[-1].replace('.', '-')[:20]
#     job_id = f"{safe_model_name}-{request_id[:15]}-{job_timestamp_suffix}"[:63].lower()

#     batch_client = batch_v1.BatchServiceClient()

#     full_dataset_gcs_path = f"{new_data_bucket.rstrip('/')}/{dataset_path.lstrip('/')}"
#     output_dir_for_job = f"{new_model_output_bucket.rstrip('/')}/model_outputs/{model_name.replace('/', '_')}/{request_id}"

#     # --- Environment Variables for the container ---
#     env_vars = {
#         "DATASET": full_dataset_gcs_path,
#         "OUTPUT_DIR": output_dir_for_job,
#         "MODEL_NAME": model_name,
#         "EPOCHS": str(epochs),
#         "LEARNING_RATE": str(learning_rate),
#         "LORA_RANK": str(lora_rank),
#         "REQUEST_ID": request_id,
#         "PROJECT_ID": gcp_project_id # Pass the hardcoded project_id to the container
#     }
#     if hf_token_runtime:
#         env_vars["HF_TOKEN"] = hf_token_runtime

#     # --- Runnable: Defines the container to run ---
#     runnable = batch_v1.types.Runnable()
#     runnable.container = batch_v1.types.Runnable.Container(image_uri=image_uri)
#     runnable.environment = batch_v1.types.Environment(variables=env_vars)

#     # --- ComputeResource: CPU, Memory, GPU ---
#     compute_resource_config = batch_v1.types.ComputeResource(
#         cpu_milli=cpu_milli,
#         memory_mib=memory_mib,
#         boot_disk_mib=boot_disk_mib
#     )

#     # --- TaskSpec: Defines a single task ---
#     task = batch_v1.types.TaskSpec(
#         runnables=[runnable],
#         compute_resource=compute_resource_config,
#         max_run_duration=Duration(seconds=max_run_duration_seconds)
#     )

#     # --- TaskGroup: A group of identical tasks ---
#     group = batch_v1.types.TaskGroup(task_count=1, task_spec=task)

#     # --- AllocationPolicy: How VMs are provisioned ---
#     instance_policy = batch_v1.types.AllocationPolicy.InstancePolicy(
#     machine_type=machine_type,
#     # accelerators=[
#     #     batch_v1.types.AllocationPolicy.Accelerator(
#     #         type_="nvidia-l4",
#     #         count=1
#     #     )
#     # ]
#     )
#     allocation_policy_config = batch_v1.types.AllocationPolicy(
#         instances=[batch_v1.types.AllocationPolicy.InstancePolicyOrTemplate(policy=instance_policy)]
#     )
#     if batch_job_sa_email:
#         allocation_policy_config.service_account = batch_v1.types.ServiceAccount(email=batch_job_sa_email)
    
#     # --- Job: The overall Batch job definition ---
#     job_definition = batch_v1.types.Job(
#         task_groups=[group],
#         allocation_policy=allocation_policy_config,
#         labels={"job-type": "finetuning", "model-id": model_name.replace("/", "-")[:30], "req-id": request_id[:20]},
#         logs_policy=batch_v1.types.LogsPolicy(
#             destination=batch_v1.types.LogsPolicy.Destination.CLOUD_LOGGING
#         )
#     )

#     # --- CreateJobRequest ---
#     create_request = batch_v1.types.CreateJobRequest(
#         parent=f"projects/{gcp_project_id}/locations/{gcp_region}",
#         job_id=job_id,
#         job=job_definition,
#     )

#     try:
#         created_job = batch_client.create_job(request=create_request)
#         return created_job
#     except Exception as e:
#         raise

# def submit_finetuning_job(
#     model_name: str,
#     dataset_path: str,
#     epochs: int,
#     learning_rate: float,
#     lora_rank: int = 4,
#     request_id: str = None):

#     output_dir = f"{NEW_MODEL_OUTPUT_BUCKET}/model/{request_id}"
#     project_id= VERTEX_AI_PROJECT
#     url = "https://llm-garage-finetune-513913820596.us-central1.run.app/run-finetune-job"
#     payload = {
#     "dataset": f"{NEW_DATA_BUCKET}/{dataset_path}",
#     "output_dir": output_dir,
#     "model_name": model_name,
#     "epochs": epochs,
#     "learning_rate": learning_rate,
#     "lora_rank": lora_rank,
#     "request_id": request_id,
#     "project_id": project_id
#     }
    
#     response = requests.post(url, json=payload)

#     if response.status_code == 200:
#         print("Job submitted successfully:", response.json())
#     else:
#         print("Failed to submit job:", response.status_code, response.text)


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
    since_timestamp: datetime | None,
    project_id: str = VERTEX_AI_PROJECT, # Use from env var
    limit: int = 200 
):
    client = logging.Client(project=project_id)

    # Define the convention for the custom log name
    # This MUST match the log name used in your gemma-garage-finetuning/src/finetuning.py script
    custom_log_name = f"gemma_garage_job_logs_{request_id}"

    # Build filter string based on whether since_timestamp is provided
    if since_timestamp is not None:
        if since_timestamp.tzinfo is None:
            since_timestamp = since_timestamp.replace(tzinfo=timezone.utc)
        
        filter_str = (
            f'logName="projects/{project_id}/logs/{custom_log_name}" '
            f'AND timestamp >= "{since_timestamp.isoformat()}"'
        )
    else:
        # Fetch all logs when no since_timestamp is provided
        filter_str = f'logName="projects/{project_id}/logs/{custom_log_name}"'
    
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

#  Function to parse logs and extract loss values
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