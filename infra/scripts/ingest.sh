#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

#  Activate venv if present 
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

#  Load .env 
if [ -f ".env" ]; then
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-8000}"
API_URL="http://${APP_HOST}:${APP_PORT}/ingest"

#  Usage 
usage() {
    echo "Usage: $0 <file_or_directory> [file_or_directory ...]"
    echo ""
    echo "  Uploads one or more files (or all supported files in a directory)"
    echo "  to the running RAG ingest API at ${API_URL}"
    echo ""
    echo "  Supported extensions: .pdf .txt .md .docx"
    exit 1
}

[ $# -eq 0 ] && usage

SUPPORTED_EXTS=("pdf" "txt" "md" "docx")

ingest_file() {
    local file="$1"
    local ext="${file##*.}"
    local valid=false

    for e in "${SUPPORTED_EXTS[@]}"; do
        [ "$ext" = "$e" ] && valid=true && break
    done

    if [ "$valid" = false ]; then
        echo "  [skip] Unsupported extension: $file"
        return
    fi

    echo "  [upload] $file"
    response=$(curl -s -o /tmp/ingest_response.json -w "%{http_code}" \
        -X POST "$API_URL" \
        -F "file=@${file}")

    if [ "$response" = "200" ]; then
        chunks=$(python3 -c "import json,sys; d=json.load(open('/tmp/ingest_response.json')); print(d.get('chunks_ingested',0))" 2>/dev/null || echo "?")
        echo "  [ok]     $file → ${chunks} chunk(s) ingested"
    else
        echo "  [error]  $file → HTTP ${response}"
        cat /tmp/ingest_response.json 2>/dev/null || true
    fi
}

#  Process arguments 
for target in "$@"; do
    if [ -f "$target" ]; then
        ingest_file "$target"
    elif [ -d "$target" ]; then
        echo "==> [ingest] Scanning directory: $target"
        while IFS= read -r -d '' file; do
            ingest_file "$file"
        done < <(find "$target" -type f \( -name "*.pdf" -o -name "*.txt" -o -name "*.md" -o -name "*.docx" \) -print0)
    else
        echo "  [warn] Not found: $target"
    fi
done

echo ""
echo "✅  Ingest complete."