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

# Setup

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

    echo "==> Creating directories"
    mkdir -p data/uploads
    mkdir -p data/chroma

    echo "==> Downloading embedding model"
    python3 - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
PY

    if command -v ollama &>/dev/null; then
        echo "==> Pulling tinyllama model"
        ollama pull tinyllama
    else
        echo "⚠ Install Ollama: https://ollama.com"
    fi

    echo "✅ Setup complete"
}

# Start API + UI

cmd_start() {

    load_env
    activate_venv

    echo "[run] Starting FastAPI http://${APP_HOST}:${APP_PORT}"

    uvicorn app.main:app \
        --host "$APP_HOST" \
        --port "$APP_PORT" &

    API_PID=$!

    echo "[run] Starting Gradio http://0.0.0.0:${UI_PORT}"

    python -m app.ui &

    UI_PID=$!

    trap 'echo ""; echo "[run] stopping..."; kill $API_PID $UI_PID 2>/dev/null; exit 0' INT TERM

    echo "[run] running (Ctrl+C to stop)"

    wait
}

# Chat helper

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

# Ingest helper

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

# CLI

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
    echo "bash run.sh setup"
    echo "bash run.sh start"
    echo "bash run.sh chat \"question\""
    echo "bash run.sh ingest file.txt"
    echo ""
    ;;

*)
    echo "Unknown command"
    exit 1
    ;;

esac