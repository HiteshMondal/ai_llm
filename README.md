# AI RAG App

A local **Retrieval-Augmented Generation (RAG)** application built with **FastAPI**, **LangChain**, and **Gradio**. Supports pluggable LLM backends, semantic search via ChromaDB, and multiple data source connectors.

---

## Features

- 📄 Ingest PDF, TXT, Markdown, and DOCX files
- 🔍 Semantic search via ChromaDB vector store with query expansion
- 🤖 Pluggable LLM backends — Ollama, Gemini, Groq, OpenRouter, OpenAI
- 🌐 REST API (FastAPI) + Web UI (Gradio)
- 🔗 Data source connectors — Web URLs, GitHub, Notion, Google Drive
- ⚡ Query caching with TTL, optional cross-encoder re-ranking
- 💬 Multi-turn session memory
- 🐳 Docker Compose support

---

## Project Structure
.
├── app/
│   ├── api.py          # FastAPI routes (chat, ingest, manage, sources)
│   ├── config.py       # Settings and logger
│   ├── main.py         # App entrypoint and lifespan
│   ├── rag.py          # RAG pipeline (retrieval, LLM, cache)
│   ├── ui.py           # Gradio web interface
│   ├── connectors/     # Web, GitHub, Notion, Google Drive
│   └── requirements.txt
├── data/
│   ├── uploads/        # Uploaded and default documents
│   └── embeddings/     # ChromaDB vector store
├── run.sh              # Setup and start script
└── README.md

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) — for local LLM inference *(optional if using API providers)*
- Docker & Docker Compose *(optional)*

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/HiteshMondal/ai_llm.git
cd ai_llm
```

### 2. Configure *(optional)*

Create a `.env` file in the project root:

```env
# Switch LLM provider (default: ollama)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash-lite
GEMINI_API_KEY=your_key_here

# Or use Groq (free tier available)
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here

# Or OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

### 3. Start

```bash
bash run.sh
```

This will automatically:
- Install system dependencies
- Create a Python virtual environment
- Install all Python packages
- Download the local embedding model
- Pull the default Ollama model *(if using Ollama)*
- Start FastAPI and Gradio

| Service   | URL                          |
|-----------|------------------------------|
| Gradio UI | http://localhost:7860        |
| FastAPI   | http://localhost:8000        |
| API Docs  | http://localhost:8000/docs   |

### 4. Ingest documents

Upload files via the **Gradio UI → Upload tab**, or use the API:

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/document.pdf"
```

Or drop a `default_knowledge.txt` in `data/uploads/` — it will be auto-ingested on startup if the vector DB is empty.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest` | Upload and ingest a document |
| POST | `/chat` | Ask a question (RAG, non-streaming) |
| POST | `/chat/stream` | Ask a question (streaming SSE) |
| GET | `/documents` | List all ingested documents |
| GET | `/documents/{source}/preview` | Preview a document's chunks |
| DELETE | `/documents/{source}` | Delete a document |
| POST | `/sources/ingest/web` | Ingest from URLs |
| POST | `/sources/ingest/github` | Ingest from GitHub repos |
| POST | `/sources/ingest/notion` | Ingest from Notion pages |
| POST | `/sources/ingest/gdrive` | Ingest from Google Drive |

**Chat example:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the document", "k": 4}'
```

**Streaming chat example:**

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain the architecture", "provider": "gemini"}'
```

---

## Supported LLM Providers

| Provider | Default Model | Requires |
|----------|--------------|---------|
| `ollama` | `tinyllama` | Ollama installed locally |
| `gemini` | `gemini-2.0-flash-lite` | `GEMINI_API_KEY` |
| `groq` | `llama3-8b-8192` | `GROQ_API_KEY` |
| `openrouter` | `mistralai/mistral-7b-instruct:free` | `OPENROUTER_API_KEY` |
| `openai` | `gpt-3.5-turbo` | `OPENAI_API_KEY` |

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

All settings can be overridden via `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM backend provider |
| `LLM_MODEL` | `tinyllama` | Model name |
| `LLM_TEMPERATURE` | `0.2` | Response randomness (0 = deterministic) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | `700` | Characters per document chunk |
| `CHUNK_OVERLAP` | `120` | Overlap between chunks |
| `RERANKER_ENABLED` | `false` | Enable cross-encoder re-ranking |
| `RERANKER_TOP_N` | `5` | Chunks passed to LLM after re-ranking |
| `QUERY_CACHE_TTL_SECONDS` | `600` | Cache expiry in seconds |
| `SESSION_MEMORY_TURNS` | `6` | Chat history turns sent to LLM |
| `APP_PORT` | `8000` | FastAPI port |
| `UI_PORT` | `7860` | Gradio port |

---

## Docker

```bash
cd app/docker
docker compose up --build
```