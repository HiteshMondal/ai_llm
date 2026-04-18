#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-8000}"
UI_PORT="${UI_PORT:-7860}"

SUPPORTED_EXTS=("txt" "md")

# Helpers

activate_venv() {
    source .venv/bin/activate
}

load_env() {
    if [ -f ".env" ]; then
        set -o allexport
        source .env
        set +o allexport
    fi
}

encode_json_string() {
    printf '%s' "$1" \
        | python3 -c "import sys,json;print(json.dumps(sys.stdin.read()))"
}

install_ollama() {

    if command -v ollama &>/dev/null; then
        echo "[run] Ollama already installed"
        return
    fi

    echo "[run] Installing Ollama"
    curl -fsSL https://ollama.com/install.sh | sh
}

ensure_setup_done() {

    if [ ! -d ".venv" ]; then
        echo ""
        echo "==> First run detected — performing setup"
        echo ""
        cmd_setup
    fi
}

# Setup
cmd_setup() {

    echo "==> Creating virtual environment"

    python3 -m venv .venv

    activate_venv

    pip install --upgrade pip

    echo "==> Installing CPU PyTorch"

    pip install torch \
        --index-url https://download.pytorch.org/whl/cpu

    echo "==> Installing dependencies"

    pip install -r app/requirements.txt

    echo "==> Installing connectors + providers"

    pip install \
        google-auth-oauthlib \
        google-auth-httplib2 \
        google-api-python-client \
        notion-client \
        trafilatura \
        PyGithub \
        langchain-google-genai \
        langchain-groq \
        langchain-openai

    mkdir -p data/uploads
    mkdir -p data/chroma
    mkdir -p data/embeddings

    load_env

    EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-local}"
    LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
    RERANKER_ENABLED="${RERANKER_ENABLED:-false}"

    if [ "$EMBEDDING_PROVIDER" = "local" ]; then

        echo "==> Downloading embedding model"

        python3 <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
EOF

    fi

    if [ "$RERANKER_ENABLED" = "true" ]; then

        echo "==> Downloading reranker"

        python3 <<EOF
from sentence_transformers import CrossEncoder
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
EOF

    fi

    if [ "$LLM_PROVIDER" = "ollama" ]; then

        install_ollama

        echo "==> Pulling tinyllama"

        ollama serve &>/dev/null &
        OLLAMA_PID=$!

        sleep 3

        ollama pull tinyllama

        kill "$OLLAMA_PID" 2>/dev/null || true

    fi

    echo "✅ Setup finished"
}

# Start services
cmd_start() {

    load_env

    activate_venv

    LLM_PROVIDER="${LLM_PROVIDER:-ollama}"

    if [ "$LLM_PROVIDER" = "ollama" ]; then

        install_ollama

        if ! curl -sf http://localhost:11434 &>/dev/null; then

            echo "[run] Starting Ollama"

            ollama serve &>/dev/null &
            OLLAMA_PID=$!

            sleep 3
        fi

    fi
    echo "==================================================="
    echo "[run] Starting FastAPI → http://${APP_HOST}:${APP_PORT}"
    echo "==================================================="

    uvicorn app.main:app \
        --host "$APP_HOST" \
        --port "$APP_PORT" &

    API_PID=$!
    echo "==================================================="
    echo "[run] Starting Gradio → http://localhost:${UI_PORT}"
    echo "==================================================="
    python -m app.ui &

    UI_PID=$!

    trap 'echo ""; echo "[run] stopping"; kill $API_PID $UI_PID ${OLLAMA_PID:-} 2>/dev/null; exit 0' INT TERM

    echo "[run] running (Ctrl+C to stop)"

    wait
}

# CLI Router

case "${1:-auto}" in

auto)

    ensure_setup_done
    cmd_start
    ;;

setup)

    cmd_setup
    ;;

start)

    ensure_setup_done
    cmd_start
    ;;

*)
    echo ""
    echo "Usage:"
    echo "  ./run.sh              Auto setup + start"
    echo "  ./run.sh setup        Install dependencies"
    echo "  ./run.sh start        Start services"
    echo ""
    ;;
esac