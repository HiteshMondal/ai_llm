# AI LLM RAG App

A local Retrieval-Augmented Generation (RAG) application built with **FastAPI**, **LangChain**, and **Gradio**. Supports pluggable LLM and embedding backends (Ollama, OpenAI, HuggingFace) and a ChromaDB vector store.

---

## Features

- 📄 Ingest PDF, TXT, Markdown, and DOCX files
- 🔍 Semantic search via ChromaDB vector store
- 🤖 Pluggable LLM backends — Ollama, OpenAI, HuggingFace
- 🌐 REST API (FastAPI) + Web UI (Gradio)
- 🐳 Docker Compose support
- 🏋️ Training & LoRA fine-tuning scripts included

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) — for local LLM and embeddings
- Docker & Docker Compose *(optional — for containerised run)*

---

## Project Structure

```
ai_llm/
├── run.sh                  # ← single entry point for all commands
├── config.yaml             # central configuration
├── .env.example            # environment variable template
├── requirements.txt
│
├── app/
│   ├── api/                # FastAPI route handlers (chat, ingest, health)
│   ├── core/               # embedder, LLM engine, RAG pipeline
│   ├── ingestion/          # document loaders, cleaner, chunker
│   ├── ui/                 # Gradio web interface
│   ├── utils/              # config, logger, helpers
│   └── main.py             # FastAPI app entry point
│
├── data/
│   ├── uploads/            # files uploaded via API or UI
│   ├── processed/          # post-ingestion copies
│   ├── raw/                # source documents (put files here)
│   ├── embeddings/         # ChromaDB persistence
│   └── cache/
│
├── infra/
│   ├── docker/             # Dockerfile + docker-compose.yml
│   └── scripts/            # setup.sh, start.sh, ingest.sh
│
├── models/
│   ├── base/
│   ├── embeddings/
│   ├── fine-tuned/
│   └── quantized/
│
└── training/
    ├── configs/            # train_config.yaml, fine_tune_config.yaml, eval_config.yaml
    ├── datasets/           # training data (.txt, .jsonl)
    ├── scripts/            # train.py, fine_tune.py, evaluate.py
    ├── checkpoints/
    └── logs/
```

---

## Quick Start

### 1. Clone

```bash
git clone <repo-url> ai_llm
cd ai_llm
```

### 2. Setup

```bash
bash run.sh setup
```

This will:
- Create all required directories
- Copy `.env.example` → `.env`
- Create a Python virtual environment and install all dependencies
- Pull default Ollama models (`llama3.2`, `nomic-embed-text`)

### 3. Configure *(optional)*

Edit `.env` or `config.yaml` to change the LLM backend, ports, chunk size, etc.

```bash
# Example: switch to OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

### 4. Start

```bash
bash run.sh start
```

| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Gradio UI | http://localhost:7860 |

### 5. Ingest documents

Put your files in `data/raw/`, then:

```bash
bash run.sh ingest data/raw/
```

Or ingest a single file:

```bash
bash run.sh ingest /path/to/document.pdf
```

### 6. Chat

```bash
bash run.sh chat "What is this document about?"
```

Or open the Gradio UI at http://localhost:7860.

---

## run.sh — All Commands

```bash
bash run.sh setup                      # first-time setup
bash run.sh start                      # start FastAPI + Gradio UI
bash run.sh start --api-only           # start FastAPI only
bash run.sh start --docker             # start via Docker Compose

bash run.sh ingest data/raw/           # ingest a directory
bash run.sh ingest /path/to/file.pdf   # ingest a single file
bash run.sh chat "your question"       # one-off question via API

bash run.sh health                     # check API health status

bash run.sh train                      # run training script
bash run.sh finetune                   # run LoRA fine-tuning
bash run.sh evaluate                   # run evaluation (perplexity + BLEU)

bash run.sh stop                       # stop Docker Compose services
bash run.sh logs                       # tail Docker Compose logs
bash run.sh help                       # show all commands
```

---

## Docker

```bash
bash run.sh start --docker
```

Or manually:

```bash
cd infra/docker
docker compose up --build
```

Starts three containers: `ai_llm_app` (FastAPI + Gradio), `ai_llm_ollama`, and `ai_llm_chromadb`.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + current config |
| POST | `/ingest` | Upload and ingest a document |
| GET | `/ingest/list` | List uploaded files |
| POST | `/chat` | Ask a question (RAG) |

Full interactive docs: http://localhost:8000/docs

**Example — ingest via curl:**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/document.pdf"
```

**Example — chat via curl:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the document", "k": 4}'
```

---

## Training & Fine-tuning

Edit the configs in `training/configs/` before running.

```bash
bash run.sh train        # pre-train on plain .txt data
bash run.sh finetune     # LoRA fine-tune on instruction .jsonl data
bash run.sh evaluate     # evaluate with perplexity + BLEU
```

Dataset formats:
- **Training:** `training/datasets/train.txt` — plain text, one document per line
- **Fine-tuning:** `training/datasets/finetune.jsonl` — `{"instruction": "...", "response": "..."}`
- **Evaluation:** `training/datasets/eval.jsonl` — same format as fine-tuning

---

## Supported File Types

| Extension | Format |
|-----------|--------|
| `.pdf` | PDF documents |
| `.txt` | Plain text |
| `.md` | Markdown |
| `.docx` | Word documents |

---

## Configuration Reference

Key settings in `config.yaml` (all overridable via `.env`):

| Setting | Default | Description |
|---------|---------|-------------|
| `llm.provider` | `ollama` | `ollama` / `openai` / `huggingface` |
| `llm.model` | `llama3.2` | Model name for the LLM |
| `embeddings.provider` | `ollama` | `ollama` / `openai` / `sentence-transformers` |
| `embeddings.model` | `nomic-embed-text` | Embedding model name |
| `ingestion.chunk_size` | `512` | Token size per chunk |
| `ingestion.chunk_overlap` | `64` | Overlap between chunks |
| `app.port` | `8000` | FastAPI port |