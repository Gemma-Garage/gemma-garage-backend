import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints import model, dataset, finetune, download, inference, ingest, huggingface

app = FastAPI(title="LLM Garage API")

# Allow multiple origins for CORS, can be overridden with environment variable
default_origins = [
    "https://gemma-garage.web.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=default_origins,  # Use specific origins when credentials are enabled
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to add official Hugging Face OAuth endpoints if in HF Space environment
try:
    from huggingface_hub._oauth import attach_huggingface_oauth
    
    # Check if we're in a HuggingFace Space environment
    if os.getenv("SPACE_ID") or os.getenv("HF_HUB_ENDPOINT"):
        attach_huggingface_oauth(app)
        print("HuggingFace OAuth endpoints attached successfully (HF Space environment)")
    else:
        print("Not in HF Space environment - OAuth endpoints not attached")
        print("Using manual OAuth implementation instead")
except Exception as e:
    print(f"Could not attach HF OAuth endpoints: {e}")
    print("Using manual OAuth implementation instead")

app.include_router(model.router, prefix="/model", tags=["Model"])
app.include_router(dataset.router, prefix="/dataset", tags=["Dataset"])
app.include_router(finetune.router, prefix="/finetune", tags=["Fine-tuning"])
app.include_router(download.router, prefix="/download", tags=["Download"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingest"])
app.include_router(huggingface.router, prefix="/huggingface", tags=["Hugging Face"])
# OAuth endpoints are also available under /oauth/huggingface for specific OAuth calls
app.include_router(huggingface.router, prefix="/oauth/huggingface", tags=["Hugging Face OAuth"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
