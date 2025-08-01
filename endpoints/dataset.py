from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
import os
import json
import csv
import re
import unicodedata
from utils.file_handler import save_uploaded_file
import google.generativeai as genai
from google.cloud import storage
import uuid
import requests
from datasets import load_dataset
import tempfile
from huggingface_hub import HfApi

router = APIRouter()

NEW_DATA_BUCKET = os.environ.get("NEW_DATA_BUCKET", "gs://default-data-bucket")  # Provide a sensible default or raise an error if not set

#pydantic model for the request body
class AugmentRequest(BaseModel):
    dataset_gcs_path: str
    fine_tuning_task_prompt: str
    model_choice: str = "gemini-2.5-flash-preview-05-20"
    qa_pairs: int = 50

class HFDatasetImportRequest(BaseModel):
    dataset_name: str
    split: str = "train"


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    # Save the original file to GCS
    print(f"Bucket name {NEW_DATA_BUCKET}")
    file_location = await save_uploaded_file(file, NEW_DATA_BUCKET)
    return {
        "message": f"{os.path.splitext(file.filename)[1].upper()[1:]} file uploaded successfully",
        "file_location": file_location
    }

@router.post("/import-hf-dataset")
async def import_hf_dataset(request: HFDatasetImportRequest):
    """
    Import a dataset from Hugging Face and save it to GCS.
    """
    try:
        print(f"Importing HF dataset: {request.dataset_name}, split: {request.split}")
        print(f"Dataset name type: {type(request.dataset_name)}")
        print(f"Dataset name length: {len(request.dataset_name)}")
        
        # Load the dataset from Hugging Face
        dataset = load_dataset(request.dataset_name, split=request.split)
        
        # Convert to list for JSON serialization
        dataset_list = list(dataset)
        
        # Generate a unique filename
        dataset_filename = f"hf_import_{uuid.uuid4()}.json"
        
        # Save to GCS
        storage_client = storage.Client()
        bucket_name = NEW_DATA_BUCKET.replace("gs://", "")
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(dataset_filename)
        
        # Upload the dataset as JSON
        blob.upload_from_string(
            json.dumps(dataset_list, indent=2),
            content_type='application/json'
        )
        
        file_location = f"gs://{bucket_name}/{dataset_filename}"
        
        return {
            "message": f"Successfully imported {request.dataset_name} ({request.split} split)",
            "file_location": file_location,
            "dataset_info": {
                "name": request.dataset_name,
                "split": request.split,
                "num_examples": len(dataset_list)
            }
        }
        
    except Exception as e:
        print(f"Error importing HF dataset: {str(e)}")
        error_message = str(e)
        
        # Provide more specific error messages
        if "Couldn't find any data file" in error_message:
            raise HTTPException(
                status_code=400, 
                detail=f"Dataset '{request.dataset_name}' not found. Please check the dataset name and ensure it exists on Hugging Face Hub."
            )
        elif "split" in error_message.lower():
            raise HTTPException(
                status_code=400, 
                detail=f"Split '{request.split}' not available for dataset '{request.dataset_name}'. Try a different split (train, validation, test)."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Failed to import Hugging Face dataset: {error_message}")


# Helper function to parse JSON stream and ignore broken tail (from notebook)
# This might be better placed in a utility file if used elsewhere.
def parse_json_stream_and_ignore_broken_tail(json_string_content: str) -> list:
    # ... (Implementation from the notebook, ensure logging is adapted or removed if not needed here)
    # For brevity, I'll assume this function is defined as in the notebook.
    # Make sure to handle logging appropriately for a backend service.
    # Simplified version for now:
    parsed_objects = []
    decoder = json.JSONDecoder()
    content_to_parse = json_string_content.strip()
    if content_to_parse.startswith('['):
        content_to_parse = content_to_parse[1:]
    if content_to_parse.endswith(']'):
        content_to_parse = content_to_parse[:-1]
    content_to_parse = content_to_parse.strip()
    idx = 0
    while idx < len(content_to_parse):
        start_idx = idx
        while start_idx < len(content_to_parse) and \
              (content_to_parse[start_idx].isspace() or content_to_parse[start_idx] == ','):
            start_idx += 1
        if start_idx >= len(content_to_parse):
            break
        try:
            obj, end_idx_offset = decoder.raw_decode(content_to_parse, start_idx)
            parsed_objects.append(obj)
            idx = end_idx_offset
        except json.JSONDecodeError:
            break # Stop parsing on the first error
    return parsed_objects

async def generate_synthetic_dataset_with_gemini(
    original_dataset_extract: str,
    fine_tuning_task_prompt: str,
    num_examples_to_generate: int = 50, # Default, can be parameterized
    model_name: str = "gemini-2.5-flash-preview-05-20"
):
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    print(f"DEBUG: GEMINI_API_KEY: {gemini_api_key}") # For debugging, remove in production
    
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    genai.configure(api_key=gemini_api_key)

    prompt = f"""You are an expert data generator. Your task is to create a synthetic dataset for fine-tuning a language model.
    The goal of the fine-tuned model is: {fine_tuning_task_prompt}

    Here is an extract from an example dataset that shows the desired format and style:
    --- BEGIN EXAMPLE EXTRACT ---
    {original_dataset_extract}
    --- END EXAMPLE EXTRACT ---

    Please generate {num_examples_to_generate} new examples in the same JSON format as the extract provided.
    Ensure each example is a complete JSON object. Output the examples as a JSON array.
    Start the JSON array with the tag <json_dataset> and do not write this tag anywhere else.
    Make sure you don't leave any json unfinished. The dataset json should contain a key "text", and 
    follow a chat template in which user and model messages are separated by a '<start_of_turn>user\n', '<start_of_turn>model\n', and '<end_of_turn>model\n', and '<end_of_turn>\n' tags.
    Replace Instruction: with <start_of_turn>user\n and Output (or response, or equivalent): with <start_of_turn>\nmodel. Finish each turn with <end_of_turn>.
    """

    generation_config = genai.types.GenerationConfig(
        # temperature=0.7, # Adjust as needed
        # top_p=0.95,
        # top_k=40,
        max_output_tokens=8192, # Max for gemini-1.5-flash, adjust if model changes
    )

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    model = genai.GenerativeModel(model_name=model_name, 
                                  safety_settings=safety_settings,
                                  generation_config=generation_config)

    # Using a loop for retries or generating in batches if needed (as in notebook)
    # For now, a single attempt for simplicity, but the notebook's loop is more robust.
    # The notebook also had a time.sleep(10) which might be necessary for rate limits.
    
    # print(f"DEBUG: Gemini Prompt:\n{prompt}") # For debugging

    try:
        response = await model.generate_content_async(prompt) # Use async version
        # print(f"DEBUG: Gemini Response Text:\n{response.text}") # For debugging
        
        if not response.text or "<json_dataset>" not in response.text:
            # print(f"DEBUG: Gemini full response object: {response}") # More detailed debugging
            # Check for prompt feedback if available
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                raise HTTPException(status_code=400, detail=f"Content generation blocked by API. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}")
            raise HTTPException(status_code=500, detail="Failed to generate dataset: No valid JSON found in response or missing <json_dataset> tag.")

        json_to_parse = response.text.split("<json_dataset>", 1)[1]
        json_to_parse = json_to_parse.replace("```json", "").replace("```", "").strip()
        
        # print(f"DEBUG: JSON to parse:\n{json_to_parse}") # For debugging

        dataset = parse_json_stream_and_ignore_broken_tail(json_to_parse)
        return dataset
    except Exception as e:
        # print(f"Error during Gemini API call or parsing: {str(e)}") # For debugging
        # Check if it's an HTTPException from a block reason, if so, re-raise
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error generating synthetic dataset: {str(e)}")

@router.post("/augment-gemma")
async def augment_dataset_gemma(request: AugmentRequest):

    #get parameters from request
    # Handle both full GCS paths and relative paths
    if request.dataset_gcs_path.startswith("gs://"):
        dataset_gcs_path = request.dataset_gcs_path
    else:
        dataset_gcs_path = f"{NEW_DATA_BUCKET}/{request.dataset_gcs_path}"
    
    fine_tuning_task_prompt = request.fine_tuning_task_prompt
    model_choice = request.model_choice
    num_examples_to_generate = request.qa_pairs
    
    if not dataset_gcs_path.startswith("gs://"):
        raise HTTPException(status_code=400, detail="Invalid GCS path for dataset.")
    if not fine_tuning_task_prompt.strip():
        raise HTTPException(status_code=400, detail="Fine-tuning task prompt cannot be empty.")

    storage_client = storage.Client()
    
    try:
        bucket_name, blob_name = dataset_gcs_path.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download a portion of the original dataset for context
        # Limit to avoid very large files; 20k chars should be enough for context
        original_dataset_content_bytes = blob.download_as_bytes(end=20000) 
        original_dataset_extract = original_dataset_content_bytes.decode('utf-8', errors='ignore')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read original dataset from GCS: {str(e)}")

    try:
        augmented_data = await generate_synthetic_dataset_with_gemini(
            original_dataset_extract=original_dataset_extract,
            fine_tuning_task_prompt=fine_tuning_task_prompt,
            num_examples_to_generate=num_examples_to_generate
        )
    except HTTPException as e: # Catch HTTPExceptions from generate_synthetic_dataset_with_gemini
        raise e # Re-raise them as they are already well-formed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate augmented data: {str(e)}")

    if not augmented_data:
        raise HTTPException(status_code=500, detail="Augmented data generation resulted in an empty dataset.")

    # Save the full augmented dataset to GCS
    try:
        augmented_blob_name = f"augmented/{uuid.uuid4()}_{os.path.basename(blob_name).replace('.json', '_augmented.json')}"
        # Ensure NEW_DATA_BUCKET is defined and accessible, e.g., from os.environ
        # For now, assuming it's the same bucket as the input for simplicity, but should be configurable
        augmented_bucket_name = NEW_DATA_BUCKET.replace("gs://", "") 
        augmented_bucket = storage_client.bucket(augmented_bucket_name)
        augmented_blob = augmented_bucket.blob(augmented_blob_name)
        
        augmented_blob.upload_from_string(
            json.dumps(augmented_data, indent=2),
            content_type='application/json'
        )
        augmented_dataset_gcs_path = f"gs://{augmented_bucket.name}/{augmented_blob.name}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save augmented dataset to GCS: {str(e)}")

    # Prepare preview (e.g., first 5 items)
    preview_limit = 15
    augmented_data_preview = augmented_data[:preview_limit]

    return {
        "message": "Dataset augmented successfully using Gemma.",
        "augmented_dataset_gcs_path": augmented_dataset_gcs_path,
        "preview_augmented_data": {
            "preview": augmented_data_preview,
            "full_count": len(augmented_data)
        }
    }


@router.get("/preview")
async def preview_uploaded_file(file_path: str = Query(..., alias="path")):
    """
    Preview the content of an uploaded file (JSON, CSV, or PDF).
    Returns a structured response with preview data and total count for pagination.
    """

    if not file_path.startswith("gs://"):
        file_path = f"gs://{NEW_DATA_BUCKET}/{file_path}"  # Ensure it starts with gs://

    print(file_path)

    storage_client = storage.Client()
    
    try:
        bucket_name, blob_name = file_path.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download the file content
        file_content = blob.download_as_text(encoding='utf-8')
        
        # Try to load as JSON
        try:
            json_content = json.loads(file_content)
            
            # Check if it's an array of objects (dataset format)
            if isinstance(json_content, list):
                # Return structured response for frontend pagination
                preview_limit = 50  # Show first 50 entries in preview
                preview_data = json_content[:preview_limit]
                return {
                    "preview": preview_data,
                    "full_count": len(json_content)
                }
            else:
                # For non-array JSON (e.g., single object), return as-is for backward compatibility
                return {
                    "preview": [json_content] if json_content else [],
                    "full_count": 1 if json_content else 0
                }
        except json.JSONDecodeError:
            # If not JSON, return as plain text (for CSV or PDF text extractions)
            # Wrap in a simple structure for consistency
            return {
                "preview": [{"content": file_content}],
                "full_count": 1
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing file: {str(e)}")

@router.post("/augment-unified")
async def augment_unified(request: AugmentRequest):
    """
    Unified augmentation endpoint: if the file is JSON, use the local Gemma augmentation logic;
    otherwise, call the remote synthetic-data-kit augmentation service. Output naming is consistent.
    Accepts qa_pairs as the number of QA pairs to generate.
    """
    # Determine file extension
    _, file_extension = os.path.splitext(request.dataset_gcs_path)
    file_extension = file_extension.lower()
    base_file_name = os.path.basename(request.dataset_gcs_path)
    output_augmented_name = base_file_name.replace('.json', '_augmented.json').replace('.pdf', '_augmented.json')

    if file_extension == '.json':
        # Use the existing Gemma augmentation logic (reuse augment_dataset_gemma)
        # Pass qa_pairs to the augmentation logic
        result = await augment_dataset_gemma(request)
        # Rename the file in GCS if needed for consistency
        storage_client = storage.Client()
        bucket_name, blob_name = result["augmented_dataset_gcs_path"].replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        # Copy to new consistent name if not already
        if not blob_name.endswith('_augmented.json'):
            new_blob_name = output_augmented_name
            new_blob = bucket.copy_blob(blob, bucket, new_blob_name)
            blob.delete()
            augmented_gcs_path = f"gs://{bucket_name}/{new_blob_name}"
        else:
            augmented_gcs_path = result["augmented_dataset_gcs_path"]
        # Load preview for frontend
        preview_data = result.get("preview_augmented_data", {})
        return {
            "augmented_dataset_gcs_path": augmented_gcs_path,
            "preview_augmented_data": preview_data
        }
    else:
        # Call remote augmentation service for non-JSON files
        remote_url = "https://llm-garage-augmentation-513913820596.us-central1.run.app/augment/"
        response = requests.post(
            remote_url,
            json={"file_name": base_file_name, "qa_pairs": request.qa_pairs, "prompt": request.fine_tuning_task_prompt},
            timeout=600
        )
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Remote augmentation failed: {response.text}")
        augmented_gcs_path = f"{NEW_DATA_BUCKET}/{output_augmented_name}"
        # Download the augmented file from GCS and adapt preview for frontend
        storage_client = storage.Client()
        bucket_name, blob_name = augmented_gcs_path.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        file_content = blob.download_as_text(encoding='utf-8')
        try:
            data = json.loads(file_content)
        except Exception:
            data = {}
        # Adapt preview for frontend
        preview = []
        full_count = 0
        summary = data.get("summary") if isinstance(data, dict) else None
        if isinstance(data, list):
            preview = data[:15]
            full_count = len(data)
        elif isinstance(data, dict):
            if "qa_pairs" in data and isinstance(data["qa_pairs"], list):
                preview = data["qa_pairs"][:15]
                full_count = len(data["qa_pairs"])
            elif "data" in data and isinstance(data["data"], list):
                preview = data["data"][:15]
                full_count = len(data["data"])
        return {
            "augmented_dataset_gcs_path": augmented_gcs_path,
            "preview_augmented_data": {
                "preview": preview,
                "full_count": full_count
            },
            **({"summary": summary} if summary else {})
        }
