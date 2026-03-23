#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> [setup] Project root: $PROJECT_ROOT"

#  1. Create directory structure 
echo "==> [setup] Creating data and model directories..."
mkdir -p \
    data/uploads \
    data/processed \
    data/raw \
    data/embeddings \
    data/cache \
    models/base \
    models/embeddings \
    models/fine-tuned \
    models/quantized \
    training/logs \
    training/checkpoints \
    training/datasets \
    training/configs

#  2. Copy .env if missing 
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "==> [setup] Created .env from .env.example — edit it before running the app."
else
    echo "==> [setup] .env already exists, skipping."
fi

#  3. Python virtual environment 
if [ ! -d ".venv" ]; then
    echo "==> [setup] Creating Python virtual environment..."
    python3 -m venv .venv
fi

echo "==> [setup] Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> [setup] Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

#  4. Check Ollama 
if command -v ollama &>/dev/null; then
    echo "==> [setup] Ollama found. Pulling default models..."
    ollama pull llama3.2       || echo "    [warn] Could not pull llama3.2"
    ollama pull nomic-embed-text || echo "    [warn] Could not pull nomic-embed-text"
else
    echo "==> [setup] Ollama not found. Install from https://ollama.com and pull your models manually."
fi

echo ""
echo "✅  Setup complete. Next steps:"
echo "    1. Edit .env if needed"
echo "    2. Run:  bash infra/scripts/start.sh"