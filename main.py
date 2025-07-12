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
