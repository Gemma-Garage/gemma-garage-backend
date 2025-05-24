from google.cloud import aiplatform
from google.cloud import logging
import ast
from datetime import datetime
import json


def run_vertexai_job(model_name, dataset_path, epochs, learning_rate, lora_rank):
    NEW_STAGING_BUCKET = "gs://llm-garage-vertex-staging" # Example, choose a unique name
    NEW_DATA_BUCKET = "gs://llm-garage-datasets"         # Example, choose a unique name
    NEW_MODEL_OUTPUT_BUCKET = "gs://llm-garage-models/gemma-peft-vertex-output" #"gs://llm-garage-models"   # Example, choose a unique name

    aiplatform.init(project="llm-garage", 
                    location="us-central1",
                    staging_bucket=NEW_STAGING_BUCKET)

    job_display_name = "gemma-peft-finetune-job-llm-garage" # Store display name in a variable

    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_display_name, # Use the variable here
        container_uri="gcr.io/llm-garage/gemma-finetune:latest", # Image from llm-garage GCR
        # model_serving_container_image_uri is for deploying the trained model, not for training itself.
        model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest", # Optional
    )

    # Arguments for your training_task.py script
    training_args = [
        f"--dataset={NEW_DATA_BUCKET}/questions.json",         # GCS path to your data in llm-garage
        f"--output_dir={NEW_MODEL_OUTPUT_BUCKET}/model/", # GCS path for output in llm-garage
        "--model_name=google/gemma-3-1b-it", # Or your desired model, updated to a valid gemma model
        "--epochs=1",                   # Example
        "--learning_rate=0.0002",       # Example
        "--lora_rank=4"                 # Example
        # Add other arguments as needed by training_task.py
    ]

    # Define the machine type and accelerators for the training job
    print(f"Submitting training job: {job_display_name}") # Use the variable for pre-submission logging
    job.run(
        base_output_dir=f"{NEW_MODEL_OUTPUT_BUCKET}", # Vertex AI specific outputs in llm-garage
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        args=training_args,
        replica_count=1,
        sync=False,
        service_account="513913820596-compute@developer.gserviceaccount.com"
        # service_account="YOUR_VERTEX_AI_CUSTOM_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com" # Optional: if needed
    )

def get_logs(
    project_id="llm-garage",
    log_name="projects/llm-garage/logs/gemma-finetune-logs",
    limit=100):
    client = logging.Client(project=project_id)

    # Query
    filter_str = f'logName="{log_name}"'
    entries = client.list_entries(
        filter_=filter_str,
        order_by=logging.DESCENDING,
        max_results=limit,
    )

    logs = []
    for entry in entries:
        logs.append({
            "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
            "textPayload": entry.payload if isinstance(entry.payload, str) else None
        })

    return extract_loss_from_logs(logs)


# 🧠 Function to parse logs and extract loss values
def extract_loss_from_logs(logs):
    loss_values = []

    for log in logs:
        text = log.get("textPayload")
        timestamp = log.get("timestamp")

        if text and "loss" in text:
            try:
                parsed = ast.literal_eval(text)

                if isinstance(parsed, dict) and "loss" in parsed:
                    loss = parsed["loss"]
                    loss_values.append((timestamp, loss))
            except (ValueError, SyntaxError):
                continue

    if not loss_values:
        return None

    # Sort by timestamp
    loss_values.sort(key=lambda x: datetime.fromisoformat(x[0].replace("Z", "+00:00")))

    # Return json
    return json.dumps([{"timestamp": t, "loss": l} for t, l in loss_values], indent=2)

# 🎯 Main function to run both steps
if __name__ == "__main__":

    logs = get_logs(
        project_id="llm-garage",
        log_name="projects/llm-garage/logs/gemma-finetune-logs",
        limit=100,
        credentials_path="path/to/your/service-account.json"  # Or None if using ADC
    )

    parsed_losses = extract_loss_from_logs(logs)

    print("Losses in chronological order:")
    for entry in parsed_losses:
        print(entry)