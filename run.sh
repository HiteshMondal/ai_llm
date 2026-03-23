#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

#  Colours 
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[run]${RESET} $*"; }
success() { echo -e "${GREEN}[run]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[run]${RESET} $*"; }
error()   { echo -e "${RED}[run]${RESET} $*" >&2; }

#  Usage 
usage() {
    echo -e "
${BOLD}Usage:${RESET}  bash run.sh <command> [options]

${BOLD}Commands:${RESET}
  ${GREEN}setup${RESET}               First-time setup (venv, deps, Ollama models)
  ${GREEN}start${RESET}               Start FastAPI + Gradio UI
  ${GREEN}start --api-only${RESET}    Start FastAPI only (no Gradio)
  ${GREEN}start --docker${RESET}      Start via Docker Compose
  ${GREEN}ingest${RESET} <path>       Ingest a file or directory of documents
  ${GREEN}chat${RESET} \"<question>\"   Ask a one-off question via the API
  ${GREEN}health${RESET}              Check API health
  ${GREEN}train${RESET}               Run training script
  ${GREEN}finetune${RESET}            Run LoRA fine-tuning script
  ${GREEN}evaluate${RESET}            Run evaluation script
  ${GREEN}stop${RESET}                Stop Docker Compose services
  ${GREEN}logs${RESET}                Tail Docker Compose logs
  ${GREEN}help${RESET}                Show this message
"
    exit 0
}

#  Helpers 
activate_venv() {
    if [ -f ".venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    else
        error "Virtual environment not found. Run:  bash run.sh setup"
        exit 1
    fi
}

load_env() {
    if [ -f ".env" ]; then
        set -o allexport
        # shellcheck disable=SC1091
        source .env
        set +o allexport
    fi
}

require_running_api() {
    local host="${APP_HOST:-127.0.0.1}"
    local port="${APP_PORT:-8000}"
    if ! curl -sf "http://${host}:${port}/health" >/dev/null 2>&1; then
        error "API is not running at http://${host}:${port}. Start it first with:  bash run.sh start"
        exit 1
    fi
}

#  Commands 

cmd_setup() {
    info "Running setup..."
    bash infra/scripts/setup.sh
}

cmd_start() {
    local mode="${1:-}"
    load_env

    if [ "$mode" = "--docker" ]; then
        info "Starting via Docker Compose..."
        docker compose -f infra/docker/docker-compose.yml up --build
        return
    fi

    activate_venv

    local host="${APP_HOST:-0.0.0.0}"
    local port="${APP_PORT:-8000}"
    local debug="${APP_DEBUG:-false}"
    local ui_port="${UI_PORT:-7860}"

    if [ "$mode" = "--api-only" ]; then
        info "Starting FastAPI at http://${host}:${port}"
        if [ "$debug" = "true" ]; then
            uvicorn app.main:app --host "$host" --port "$port" --reload
        else
            uvicorn app.main:app --host "$host" --port "$port"
        fi
        return
    fi

    # Start both FastAPI and Gradio
    info "Starting FastAPI  →  http://${host}:${port}"
    info "Starting Gradio   →  http://0.0.0.0:${ui_port}"

    if [ "$debug" = "true" ]; then
        uvicorn app.main:app --host "$host" --port "$port" --reload &
    else
        uvicorn app.main:app --host "$host" --port "$port" &
    fi
    API_PID=$!

    python -m app.ui.app_ui &
    UI_PID=$!

    trap 'echo ""; info "Shutting down..."; kill "$API_PID" "$UI_PID" 2>/dev/null; exit 0' INT TERM

    success "Both services running. Press Ctrl+C to stop."
    wait "$API_PID" "$UI_PID"
}

cmd_ingest() {
    local target="${1:-}"
    if [ -z "$target" ]; then
        error "Please provide a file or directory path."
        echo "  Example:  bash run.sh ingest data/raw/"
        exit 1
    fi
    load_env
    bash infra/scripts/ingest.sh "$target"
}

cmd_chat() {
    local question="${1:-}"
    if [ -z "$question" ]; then
        error "Please provide a question."
        echo "  Example:  bash run.sh chat \"What is this document about?\""
        exit 1
    fi
    load_env
    local host="${APP_HOST:-127.0.0.1}"
    local port="${APP_PORT:-8000}"
    require_running_api

    info "Sending question to RAG API..."
    curl -s -X POST "http://${host}:${port}/chat" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"${question}\", \"k\": 4}" | python3 -m json.tool
}

cmd_health() {
    load_env
    local host="${APP_HOST:-127.0.0.1}"
    local port="${APP_PORT:-8000}"
    info "Checking health at http://${host}:${port}/health ..."
    curl -s "http://${host}:${port}/health" | python3 -m json.tool \
        && success "API is up." \
        || error "API is not responding."
}

cmd_train() {
    activate_venv
    info "Starting training..."
    python training/scripts/train.py --config training/configs/train_config.yaml
}

cmd_finetune() {
    activate_venv
    info "Starting LoRA fine-tuning..."
    python training/scripts/fine_tune.py --config training/configs/fine_tune_config.yaml
}

cmd_evaluate() {
    activate_venv
    info "Starting evaluation..."
    python training/scripts/evaluate.py --config training/configs/eval_config.yaml
}

cmd_stop() {
    info "Stopping Docker Compose services..."
    docker compose -f infra/docker/docker-compose.yml down
    success "Services stopped."
}

cmd_logs() {
    info "Tailing Docker Compose logs (Ctrl+C to exit)..."
    docker compose -f infra/docker/docker-compose.yml logs -f
}

#  Entrypoint 
COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    setup)            cmd_setup ;;
    start)            cmd_start "${1:-}" ;;
    ingest)           cmd_ingest "${1:-}" ;;
    chat)             cmd_chat "${1:-}" ;;
    health)           cmd_health ;;
    train)            cmd_train ;;
    finetune)         cmd_finetune ;;
    evaluate)         cmd_evaluate ;;
    stop)             cmd_stop ;;
    logs)             cmd_logs ;;
    help|--help|-h)   usage ;;
    *)
        error "Unknown command: '$COMMAND'"
        usage
        ;;
esac