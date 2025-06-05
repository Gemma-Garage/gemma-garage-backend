from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
import os
import json
import csv
import fitz  # PyMuPDF
import re
import unicodedata
from utils.file_handler import save_uploaded_file
import google.generativeai as genai
from google.cloud import storage
import uuid

router = APIRouter()

NEW_DATA_BUCKET = os.environ.get("NEW_DATA_BUCKET", "gs://default-data-bucket")  # Provide a sensible default or raise an error if not set

#pydantic model for the request body
class AugmentRequest(BaseModel):
    dataset_gcs_path: str
    fine_tuning_task_prompt: str
    model_choice: str = "gemini-1.5-flash"
    num_examples_to_generate: int = 50


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    # Save the file using a utility function
    print(f"Bucket name {NEW_DATA_BUCKET}")
    file_location = await save_uploaded_file(file, NEW_DATA_BUCKET)
    
    # Get file extension
    _, file_extension = os.path.splitext(file.filename)
    file_extension = file_extension.lower()
    
    # Process based on file type
    try:
        if file_extension == '.pdf':
            json_output = process_pdf_file(file_location)
            return {"message": "PDF processed successfully", "file_location": json_output}
        elif file_extension == '.json':
            # JSON files don't need processing
            return {"message": "JSON dataset uploaded successfully", "file_location": file_location}
        elif file_extension == '.csv':
            # Process CSV file to convert it to JSON format
            json_output = process_csv_file(file_location)
            return {"message": "CSV processed successfully", "file_location": json_output}
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def process_pdf_file(pdf_path):
    """
    Process a PDF file and convert it to JSON for fine-tuning dataset
    """
    # Extract text from PDF
    extracted_json = pdf_path.replace('.pdf', '_extracted.json')
    fine_tuning_json = pdf_path.replace('.pdf', '_finetuning.json')
    
    # Extract text from PDF
    extract_text_from_pdf(pdf_path, extracted_json)
    
    # Process extracted text into fine-tuning format
    process_pdf_json(extracted_json, fine_tuning_json)
    
    return fine_tuning_json

def extract_text_from_pdf(pdf_path, output_json):
    """
    Extracts text from each page of a PDF and saves it in a JSON file.
    Each key is formatted as "page_X" where X is the page number.
    """
    doc = fitz.open(pdf_path)
    pdf_content = {}
    for page_number in range(doc.page_count):
        page = doc[page_number]
        text = page.get_text()
        pdf_content[f"page_{page_number + 1}"] = text
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(pdf_content, f, indent=2, ensure_ascii=False)
    return output_json

def remove_headers_footers(text):
    """
    Remove lines that are likely headers or footers, such as page numbers
    or lines that match "Page X" patterns.
    """
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        # Skip lines that contain only numbers
        if re.match(r'^\s*\d+\s*$', line):
            continue
        # Skip lines matching patterns like "Page 1" (case-insensitive)
        if re.match(r'^\s*Page\s+\d+\s*$', line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

def clean_special_characters(text):
    """
    Normalize unicode characters and remove non-printable characters.
    Also collapses multiple spaces/newlines.
    """
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\x20-\x7E]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_words=100):
    """
    Split text into smaller chunks if it exceeds max_words.
    Adjust max_words based on your model's input size limitations.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

def process_pdf_json(input_file, output_file, max_words_per_chunk=100):
    """
    Loads the extracted PDF JSON, cleans the text, and splits long pages into chunks.
    Then, it formats each chunk as a dictionary with a "text" key and saves the data.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        pdf_data = json.load(f)
    
    training_examples = []
    
    for key, raw_text in pdf_data.items():
        # Remove headers/footers
        cleaned_text = remove_headers_footers(raw_text)
        # Clean special characters and extra whitespace
        cleaned_text = clean_special_characters(cleaned_text)
        # Split the cleaned text into chunks if necessary
        chunks = chunk_text(cleaned_text, max_words=max_words_per_chunk)
        # Add each chunk as a separate training example
        for chunk in chunks:
            training_examples.append({"text": chunk})
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, indent=2, ensure_ascii=False)
    
    return output_file

def process_csv_file(csv_path):
    """
    Process a CSV file and convert it to JSON for fine-tuning dataset.
    Assumes that the CSV contains text data that needs to be formatted for fine-tuning.
    """
    # Output JSON path
    output_json = csv_path.replace('.csv', '_finetuning.json')
    
    training_examples = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            # Get headers
            headers = next(csv_reader, None)
            
            # If no headers, use default column names
            if not headers:
                headers = [f"column_{i}" for i in range(len(next(csv_reader)))]
                # Reset file pointer
                f.seek(0)
                
            # Read rows
            for row in csv_reader:
                if not row:  # Skip empty rows
                    continue
                
                # Combine all text fields in the row
                text = " ".join([str(value) for value in row if value.strip()])
                
                # Clean special characters
                cleaned_text = clean_special_characters(text)
                
                # Add as training example
                if cleaned_text:
                    training_examples.append({"text": cleaned_text})
    except Exception as e:
        # If CSV reading fails, try a more flexible approach with different delimiters
        training_examples = process_flexible_csv(csv_path)
    
    # Write to JSON file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, indent=2, ensure_ascii=False)
    
    return output_json

def process_flexible_csv(csv_path):
    """
    Process CSV with different delimiters if standard CSV reading fails.
    Tries different delimiters (comma, tab, semicolon) to parse the file.
    """
    delimiters = [',', '\t', ';', '|']
    training_examples = []
    
    for delimiter in delimiters:
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Try reading with this delimiter
                csv_reader = csv.reader(f, delimiter=delimiter)
                for row in csv_reader:
                    if not row:  # Skip empty rows
                        continue
                    
                    # Combine all text fields in the row
                    text = " ".join([str(value) for value in row if value.strip()])
                    
                    # Clean special characters
                    cleaned_text = clean_special_characters(text)
                    
                    # Add as training example
                    if cleaned_text:
                        training_examples.append({"text": cleaned_text})
                
                # If we got here, parsing succeeded with this delimiter
                if training_examples:
                    break
        except Exception:
            # Try next delimiter
            continue
    
    return training_examples

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
    model_name: str = "gemini-1.5-flash-latest" # Check for latest recommended model
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
    Make sure you don't leave any json unfinished.
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
    num_examples_to_generate = request.num_examples_to_generate
    
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
    preview_limit = 5
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
        raise HTTPException(status_code=400, detail="Invalid file path. Must start with gs://")

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
