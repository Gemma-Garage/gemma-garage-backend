from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
from utils.training import run_finetuning
from finetuning.finetuning import *
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket
from finetuning.finetuning import FineTuningEngine
from finetuning.vertexai import run_vertexai_job, get_logs
from utils.file_handler import UPLOAD_DIR
# import time # No longer directly used in the loop, asyncio.sleep is used
import asyncio
from datetime import datetime, timezone, timedelta # Added timedelta
import json # Added json

class FinetuneRequest(BaseModel):
    model_name: str
    dataset_path: str
    epochs: int
    learning_rate: float
    lora_rank: int = 4  # Add lora_rank with default value of 4


requests = []

router = APIRouter()

@router.websocket("/ws/train")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        payload = await websocket.receive_json()
    except Exception as e:
        await websocket.send_json({"error": "Invalid JSON payload", "details": str(e)})
        await websocket.close()
        return

    model_name = payload.get("model_name", "princeton-nlp/Sheared-LLaMA-1.3B")
    dataset_path = payload.get("dataset_path")
    epochs = payload.get("epochs")
    learning_rate = payload.get("learning_rate")
    lora_rank = payload.get("lora_rank", 4)

    await websocket.send_json({"status": "connection_success", "message": "Training parameters received.", "details": {"model_name": model_name, "dataset_path": dataset_path}})

    try:
        # Start the Vertex AI job (sync=False, so it returns quickly)
        run_vertexai_job(model_name, dataset_path, epochs, learning_rate, lora_rank)
        await websocket.send_json({"status": "job_submitted", "message": "Vertex AI training job submitted."})
    except Exception as e:
        error_message = f"Failed to submit Vertex AI job: {str(e)}"
        print(error_message)
        await websocket.send_json({"error": "Job submission failed", "details": str(e)})
        await websocket.close()
        return

    since_timestamp = datetime.now(timezone.utc)

    try:
        while True:
            new_logs_processed_this_iteration = False
            
            # Fetch logs since the last recorded timestamp
            loss_values_str = get_logs(since_timestamp) # get_logs expects a datetime object

            if loss_values_str is not None:
                # Send the raw JSON string of loss values to the client
                await websocket.send_json({"loss_values": loss_values_str})
                
                try:
                    loss_data_list = json.loads(loss_values_str)
                    if isinstance(loss_data_list, list) and loss_data_list: # Check if it's a non-empty list
                        # Logs are sorted chronologically by extract_loss_from_logs
                        last_log_entry = loss_data_list[-1]
                        latest_ts_str = last_log_entry.get("timestamp")
                        
                        if latest_ts_str:
                            # Convert ISO string to datetime object
                            # Google's ISO format might end with 'Z', replace with +00:00 for broader fromisoformat compatibility
                            dt_obj = datetime.fromisoformat(latest_ts_str.replace("Z", "+00:00"))
                            
                            # Update since_timestamp to be just after the latest processed log
                            since_timestamp = dt_obj + timedelta(microseconds=1)
                            new_logs_processed_this_iteration = True
                        else:
                            print("Warning: Last log entry missing timestamp.")
                    # If loss_data_list is empty, new_logs_processed_this_iteration remains False
                except json.JSONDecodeError as e:
                    print(f"Error decoding loss_values_str from get_logs: {e}. String was: {loss_values_str}")
                except Exception as e: # Catch other potential errors (KeyError, TypeError, etc.)
                    print(f"Error processing parsed loss_data_list: {e}")
            
            # If no new logs were successfully processed in this iteration,
            # advance since_timestamp to current time to avoid re-querying the same old range indefinitely.
            if not new_logs_processed_this_iteration:
                since_timestamp = datetime.now(timezone.utc)

            await asyncio.sleep(10) # Use asyncio.sleep in an async function

    except Exception as e:
        # Catch exceptions that occur within the WebSocket communication loop (e.g., client disconnects)
        print(f"WebSocket communication error: {e}")
        try:
            await websocket.send_json({"error": "Log streaming interrupted", "details": str(e)})
        except Exception: # Websocket might already be closed or in a bad state
            pass 
    finally:
        print("Closing WebSocket connection.")
        await websocket.close()

# @router.websocket("/ws/train")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()

#     # Wait for the client to send the JSON payload with training parameters
#     try:
#         payload = await websocket.receive_json()
#     except Exception as e:
#         await websocket.send_json({"error": "Invalid JSON payload", "details": str(e)})
#         await websocket.close()
#         return

#     # Extract the model name and other parameters from the payload, using defaults if necessary
#     model_name = payload.get("model_name", "princeton-nlp/Sheared-LLaMA-1.3B")
#     dataset_path = payload.get("dataset_path")
#     epochs = payload.get("epochs")
#     learning_rate = payload.get("learning_rate")
#     lora_rank = payload.get("lora_rank", 4)  # Get lora_rank with default of 4

#     await websocket.send_json({"test connection": "success", "model_name": model_name, "dataset_path": dataset_path})
    
#     # Get the main event loop
#     main_loop = asyncio.get_running_loop()
    
#     engine = FineTuningEngine(model_name, websocket)
#     dataset = engine.load_new_dataset(dataset_path)
#     engine.set_lora_fine_tuning(dataset, 
#                                 learning_rate=learning_rate, 
#                                 epochs=epochs,
#                                 lora_rank=lora_rank,  # Pass lora_rank to the engine
#                                 callback_loop=main_loop)  # Pass the loop to set up callbacks
    
#     # Offload the blocking training process to a thread
#     await asyncio.to_thread(engine.perform_fine_tuning)
    
#     await asyncio.sleep(1)
#     try:
#         await websocket.send_json({"status": "training complete", "weights_url": engine.weights_path})
#     except Exception as e:
#         print("Error sending final update:", e)
#     await websocket.close()

@router.post("/set_train_params")
async def set_train_params(request:FinetuneRequest):
    model_name = request.model_name
    dataset_path = request.dataset_path

    file_name = UPLOAD_DIR + "/" + dataset_path
    if not os.path.exists(UPLOAD_DIR):
        raise HTTPException(status_code=500, detail=str('Dataset not found'))

    new_request = FinetuneRequest(model_name=model_name, dataset_path=file_name)
    requests.append(new_request)
    return {"status": "success"}
    


@router.post("/")
async def finetune(request:FinetuneRequest):
    try:
        # This function should run the LoRA fine-tuning process and return a path to the saved weights.
        # weights_path = run_finetuning(model_name, dataset_path)
        # if not os.path.exists(weights_path):
        #     raise HTTPException(status_code=500, detail="Fine-tuning failed to produce weights")
        # # Return file as a download
        # return FileResponse(path=weights_path, filename=os.path.basename(weights_path))
        model_name = request.model_name
        dataset_path = request.dataset_path
        lora_rank = request.lora_rank  # Get lora_rank from request

        engine = FineTuningEngine(model_name)
        engine.load_new_dataset(dataset_path)
        engine.set_lora_fine_tuning(lora_rank=lora_rank)  # Pass lora_rank to the engine
        engine.perform_fine_tuning()
        return {"model_name": request.model_name, "dataset_path": request.dataset_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
