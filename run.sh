#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-8000}"
UI_PORT="${UI_PORT:-7860}"

SUPPORTED_EXTS=("txt" "md")

#  Helpers 

activate_venv() {
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo "[run] Virtual environment missing."
        echo "Run: bash run.sh setup"
        exit 1
    fi
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

#  Install Ollama 

install_ollama() {
    if command -v ollama &>/dev/null; then
        echo "[run] Ollama already installed: $(ollama --version)"
        return
    fi
    echo "==> Installing Ollama"
    curl -fsSL https://ollama.com/install.sh | sh
    if ! command -v ollama &>/dev/null; then
        echo "❌ Ollama installation failed"
        exit 1
    fi
    echo "✅ Ollama installed: $(ollama --version)"
}

#  Setup 

cmd_setup() {
    echo "==> Creating virtual environment"
    rm -rf .venv
    python3 -m venv .venv
    source .venv/bin/activate

    echo "==> Upgrading pip"
    pip install --upgrade pip

    echo "==> Installing CPU-only PyTorch"
    pip install torch \
        --index-url https://download.pytorch.org/whl/cpu

    echo "==> Installing dependencies"
    pip install -r requirements.txt

    echo "==> Installing connector + provider dependencies"
    pip install \
        google-auth-oauthlib google-auth-httplib2 google-api-python-client \
        notion-client trafilatura PyGithub \
        langchain-google-genai langchain-groq langchain-openai

    echo "==> Creating directories"
    mkdir -p data/uploads
    mkdir -p data/chroma
    mkdir -p data/embeddings

    # Fix: ensure connectors live under app/connectors (not root connectors/)
    if [ -d "connectors" ] && [ ! -d "app/connectors" ]; then
        echo "==> Moving connectors/ → app/connectors/"
        mv connectors app/connectors
    elif [ -d "connectors" ] && [ -d "app/connectors" ]; then
        echo "==> Root connectors/ found but app/connectors/ already exists — skipping move"
    fi

    load_env
    EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-local}"
    LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
    RERANKER_ENABLED="${RERANKER_ENABLED:-false}"

    if [ "$EMBEDDING_PROVIDER" = "local" ]; then
        echo "==> Downloading embedding model (local)"
        python3 - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
PY
    else
        echo "==> Skipping local embedding model (EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER})"
    fi

    if [ "$RERANKER_ENABLED" = "true" ]; then
        echo "==> Downloading re-ranker model"
        python3 - <<'PY'
from sentence_transformers import CrossEncoder
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
PY
    else
        echo "==> Skipping re-ranker model (set RERANKER_ENABLED=true in .env to enable)"
    fi

    if [ "$LLM_PROVIDER" = "ollama" ]; then
        install_ollama
        echo "==> Pulling tinyllama model"
        if ! curl -sf http://localhost:11434 -o /dev/null 2>&1; then
            ollama serve &>/dev/null &
            OLLAMA_SETUP_PID=$!
            sleep 3
        fi
        ollama pull tinyllama
        if [ -n "${OLLAMA_SETUP_PID:-}" ]; then
            kill "$OLLAMA_SETUP_PID" 2>/dev/null || true
        fi
    else
        echo "==> Skipping Ollama setup (LLM_PROVIDER=${LLM_PROVIDER})"
    fi

    echo "✅ Setup complete"
}

#  Start API + UI 

cmd_start() {
    load_env
    activate_venv

    # Check CPU features for Ollama
    if ! grep -qE 'avx2|avx' /proc/cpuinfo 2>/dev/null; then
        echo "⚠ CPU may lack AVX/AVX2 — Ollama models may be very slow or crash"
        echo "  Consider using a free API provider (Gemini/Groq) instead"
        echo "  Set LLM_PROVIDER=gemini and GEMINI_API_KEY in .env"
    fi

    LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
    OLLAMA_PID=""

    if [ "$LLM_PROVIDER" = "ollama" ]; then
        install_ollama
        OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
        if ! curl -sf "${OLLAMA_URL}" -o /dev/null 2>&1; then
            echo "[run] Starting Ollama server..."
            ollama serve &>/dev/null &
            OLLAMA_PID=$!
            echo "[run] Ollama PID: $OLLAMA_PID"
        else
            echo "[run] Ollama already running"
        fi
        echo "[run] Waiting for Ollama at ${OLLAMA_URL} ..."
        for i in $(seq 1 15); do
            if curl -sf "${OLLAMA_URL}" -o /dev/null 2>&1; then
                echo "[run] Ollama is up"
                break
            fi
            if [ "$i" -eq 15 ]; then
                echo "❌ Ollama did not start in time"
                exit 1
            fi
            echo "[run] waiting... ($i/15)"
            sleep 2
        done
    else
        echo "[run] Skipping Ollama (LLM_PROVIDER=${LLM_PROVIDER})"
    fi

    echo "[run] Starting FastAPI http://${APP_HOST}:${APP_PORT}"
    uvicorn app.main:app \
        --host "$APP_HOST" \
        --port "$APP_PORT" &
    API_PID=$!

    echo "[run] Starting Gradio http://0.0.0.0:${UI_PORT}"
    python -m app.ui &
    UI_PID=$!

    trap 'echo ""; echo "[run] stopping..."; kill $API_PID $UI_PID ${OLLAMA_PID:-} 2>/dev/null; exit 0' INT TERM

    echo "[run] running (Ctrl+C to stop)"
    wait
}

#  Chat helper 

cmd_chat() {
    activate_venv
    QUESTION="$1"
    JSON=$(encode_json_string "$QUESTION")
    curl -s \
        -X POST \
        "http://${APP_HOST}:${APP_PORT}/chat" \
        -H "Content-Type: application/json" \
        -d "{\"question\": $JSON, \"k\": 4}" \
        | python3 -m json.tool
}

#  Ingest helper 

cmd_ingest() {
    activate_venv
    API_URL="http://${APP_HOST}:${APP_PORT}/ingest"

    if [ $# -eq 0 ]; then
        echo "Usage:"
        echo "bash run.sh ingest <file|directory>"
        exit 1
    fi

    ingest_file() {
        FILE="$1"
        EXT="${FILE##*.}"
        VALID=false
        for e in "${SUPPORTED_EXTS[@]}"; do
            if [ "$EXT" = "$e" ]; then
                VALID=true
            fi
        done
        if [ "$VALID" = false ]; then
            echo "[skip] $FILE"
            return
        fi
        echo "[upload] $FILE"
        curl \
            -s \
            -X POST \
            "$API_URL" \
            -F "file=@${FILE}"
    }

    for TARGET in "$@"; do
        if [ -f "$TARGET" ]; then
            ingest_file "$TARGET"
        elif [ -d "$TARGET" ]; then
            find "$TARGET" \
                -type f \
                \( -name "*.txt" -o -name "*.md" \) \
                | while read -r FILE; do
                ingest_file "$FILE"
            done
        else
            echo "[warn] not found: $TARGET"
        fi
    done

    echo "✅ ingest complete"
}

#  CLI 

case "${1:-help}" in

setup)
    cmd_setup
    ;;

start)
    cmd_start
    ;;

chat)
    shift
    cmd_chat "$@"
    ;;

ingest)
    shift
    cmd_ingest "$@"
    ;;

help|--help)
    echo ""
    echo "Usage:"
    echo "  bash run.sh setup              Install dependencies and models"
    echo "  bash run.sh start              Start API + UI"
    echo "  bash run.sh chat \"question\"    Ask a question via CLI"
    echo "  bash run.sh ingest file.txt    Ingest a file or directory"
    echo ""
    echo "Environment variables (set in .env):"
    echo "  LLM_PROVIDER        ollama | gemini | groq | openrouter | openai"
    echo "  EMBEDDING_PROVIDER  local | gemini | openai"
    echo "  RERANKER_ENABLED    true | false  (default: false)"
    echo "  APP_PORT            API port      (default: 8000)"
    echo "  UI_PORT             UI port       (default: 7860)"
    echo ""
    ;;

*)
    echo "Unknown command: ${1}"
    echo "Run 'bash run.sh help' for usage."
    exit 1
    ;;

esac