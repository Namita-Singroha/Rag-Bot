FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and Ollama
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start Ollama service, pull model, and run FastAPI app
CMD ollama serve & sleep 10 && ollama pull llama3.2:3b && uvicorn app:app --host 0.0.0.0 --port 8000