FROM python:3.13-slim

#huggingface token
ARG HF_TOKEN
#Gemini API key
ARG GEMINI_KEY
# Hugging Face OAuth credentials
ARG HUGGINGFACE_CLIENT_ID
ARG HUGGINGFACE_CLIENT_SECRET
ARG FRONTEND_URL

# Set default environment variables for GCS buckets
ENV NEW_DATA_BUCKET="gs://llm-garage-datasets"
ENV NEW_MODEL_OUTPUT_BUCKET="gs://llm-garage-models/gemma-peft-vertex-output"
ENV NEW_STAGING_BUCKET="gs://llm-garage-vertex-staging"
ENV GEMINI_API_KEY="${GEMINI_KEY}"
ENV HUGGINGFACE_CLIENT_ID="${HUGGINGFACE_CLIENT_ID}"
ENV HUGGINGFACE_CLIENT_SECRET="${HUGGINGFACE_CLIENT_SECRET}"
ENV HUGGINGFACE_REDIRECT_URI="https://llm-garage-513913820596.us-central1.run.app/oauth/huggingface/callback"
ENV FRONTEND_URL="https://gemma-garage.web.app"

# Install git and other necessary tools
RUN apt-get update && \
    apt-get install -y git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and set permissions for temp directory (required for gitingest)
RUN mkdir -p /tmp && chmod 777 /tmp

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#RUN huggingface-cli login --token ${HF_TOKEN}  


# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
