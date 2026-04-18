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

## Quick Start

### 1. Clone

```bash
git clone https://github.com/HiteshMondal/ai_llm.git
cd ai_llm
```

### 2. Configure *(optional)*

Edit `.env` or `config.yaml` to change the LLM backend, ports, chunk size, etc.

```bash
# Example: switch to OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

### 3. Start

```bash
bash run.sh
```
This will:
- Create all required directories
- Copy `.env.example` → `.env`
- Create a Python virtual environment and install all dependencies
- Pull default Ollama models (`llama3.2`, `nomic-embed-text`)


| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Gradio UI | http://localhost:7860 |

### 4. Ingest documents

Put your files in `data/raw/`, then:

```bash
bash run.sh ingest data/raw/
```

Or ingest a single file:

```bash
bash run.sh ingest /path/to/document.pdf
```

Or open the Gradio UI at http://localhost:7860.

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