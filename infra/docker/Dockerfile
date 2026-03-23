FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/processed data/raw data/embeddings data/cache \
             models/base models/embeddings models/fine-tuned models/quantized \
             training/logs training/checkpoints training/datasets training/configs

EXPOSE 8000 7860

CMD ["python", "-m", "app.main"]