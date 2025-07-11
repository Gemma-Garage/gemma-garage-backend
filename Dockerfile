FROM python:3.9-slim

#huggingface token
ARG HF_TOKEN
#Gemini API key
ARG GEMINI_KEY

# Set default environment variables for GCS buckets
ENV NEW_DATA_BUCKET="gs://llm-garage-datasets"
ENV NEW_MODEL_OUTPUT_BUCKET="gs://llm-garage-models/gemma-peft-vertex-output"
ENV NEW_STAGING_BUCKET="gs://llm-garage-vertex-staging"
ENV GEMINI_API_KEY="${GEMINI_KEY}" 

# Install git and other necessary tools
RUN apt-get update && \
    apt-get install -y git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and set permissions for temp directory
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
