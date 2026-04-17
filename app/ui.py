import json
import httpx
import gradio as gr
import os
from app.config import get_settings

settings = get_settings()

BASE_URL = f"http://127.0.0.1:{settings.app_port}"

def get_client():
    return httpx.Client(
        timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10)
    )


#  Upload 

def upload_fn(files):
    if not files or len(files) == 0:
        return "No file selected."
    results = []
    for file in files:
        try:
            with open(file.name, "rb") as f:
                with get_client() as client:
                    resp = client.post(
                        f"{BASE_URL}/ingest",
                        files={"file": (os.path.basename(file.name), f)},
                        timeout=120,
                    )
            resp.raise_for_status()
            data = resp.json()
            results.append(
                f"✅ **{data['filename']}** — {data['chunks_ingested']} chunk(s) ingested."
            )
        except Exception as e:
            results.append(f"❌ Upload failed for **{file.name}**:\n`{e}`")
    return "\n\n".join(results)


#  Document Management 

def list_docs_fn():
    try:
        with get_client() as client:
            resp = client.get(f"{BASE_URL}/documents")
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        if not docs:
            return "No documents ingested yet.", gr.update(choices=[], value=None)
        lines = "\n".join(
            f"- **{d['source']}** — {d['chunk_count']} chunk(s) "
            f"[`{d.get('source_type', 'local')}`]"
            for d in docs
        )
        names = [d["source"] for d in docs]
        return lines, gr.update(choices=names, value=None)
    except Exception as e:
        return f"❌ Error: {e}", gr.update(choices=[], value=None)


def delete_doc_fn(source: str):
    source = source.strip()
    if not source:
        return "Enter a document name to delete.", gr.update()
    try:
        from urllib.parse import quote
        with get_client() as client:
            resp = client.delete(f"{BASE_URL}/documents/{quote(source)}")
        resp.raise_for_status()
        data = resp.json()
        return (
            f"✅ Deleted **{data['source']}** — {data['deleted_chunks']} chunk(s) removed.",
            gr.update(value=""),
        )
    except Exception as e:
        return f"❌ Error: {e}", gr.update()


def preview_doc_fn(source: str):
    if not source:
        return "Select a document to preview."
    try:
        from urllib.parse import quote
        with get_client() as client:
            resp = client.get(f"{BASE_URL}/documents/{quote(source)}/preview?max_chars=3000")
        resp.raise_for_status()
        data = resp.json()
        return f"### Preview: {data['source']}\n\n```\n{data['preview']}\n```"
    except Exception as e:
        return f"❌ Error: {e}"


#  Chat (streaming) 

def chat_fn(
    question: str,
    history: list,
    system_instruction: str,
    provider: str,
    model: str,
    api_key: str,
    use_stream: bool,
):
    history = history or []
    if not question.strip():
        return history, ""

    previous_history = history.copy()
    history = history + [{"role": "user", "content": question}]

    if use_stream:
        yield history + [{"role": "assistant", "content": "⏳ Thinking..."}], ""
        try:
            accumulated = ""
            sources_data = []
            json_buffer = ""
            with get_client() as client:
                with client.stream(
                    "POST",
                    f"{BASE_URL}/chat/stream",
                    json={
                        "question": question,
                        "k": 4,
                        "system_instruction": system_instruction,
                        "provider": provider,
                        "model": model,
                        "api_key": api_key,
                        "history": previous_history,
                    },
                    timeout=300,
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw = line[len("data: "):]
                        if raw == "[DONE]":
                            break
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if "error" in payload:
                            accumulated = f"❌ {payload['error']}"
                            break
                        token = payload.get("token", "")

                        # Check for sources sentinel
                        if token.strip().startswith("{"):
                            json_buffer += token
                        try:
                            maybe_json = json.loads(json_buffer)
                            if "__sources__" in maybe_json:
                                sources_data = maybe_json["__sources__"]
                                json_buffer = ""
                                continue
                        except json.JSONDecodeError:
                            pass

                        accumulated += token
                        updated = history + [{"role": "assistant", "content": accumulated}]
                        yield updated, ""

            # Append sources after streaming
            if sources_data:
                src_lines = "\n".join(
                    f"- **{s['metadata'].get('source', '?')}** — "
                    f"{s['content'][:100]}…"
                    for s in sources_data
                )
                accumulated += f"\n\n**Sources:**\n{src_lines}"

            final_history = history + [{"role": "assistant", "content": accumulated}]
            yield final_history, ""

        except httpx.TimeoutException:
            msg = "⏳ Request timed out. Try a shorter question or switch to a faster provider."
            yield history + [{"role": "assistant", "content": msg}], ""
        except Exception as e:
            yield history + [{"role": "assistant", "content": f"❌ Error: {e}"}], ""

    else:
        # Non-streaming fallback
        yield history + [{"role": "assistant", "content": "⏳ Thinking..."}], ""
        try:
            with get_client() as client:
                resp = client.post(
                    f"{BASE_URL}/chat",
                    json={
                        "question": question,
                        "k": 4,
                        "system_instruction": system_instruction,
                        "provider": provider,
                        "model": model,
                        "api_key": api_key,
                        "history": previous_history,
                    },
                )
            resp.raise_for_status()
            data = resp.json()
            answer = data["answer"]
            if data.get("cached"):
                answer = "⚡ *(cached)* " + answer
            sources = data.get("sources", [])
            if sources:
                answer += "\n\n**Sources:**\n"
                answer += "\n".join(
                    f"- **{s['metadata'].get('source', '?')}** — {s['content'][:100]}…"
                    for s in sources
                )
        except httpx.TimeoutException:
            answer = "⏳ Request timed out. Try a shorter question or switch to a faster provider."
        except Exception as e:
            answer = f"❌ Error: {e}"
        yield history + [{"role": "assistant", "content": answer}], ""


#  Connector helpers 

def web_ingest_fn(urls):
    try:
        parsed = [u.strip() for u in urls.replace(",", "\n").split("\n") if u.strip()]
        with get_client() as client:
            r = client.post(
                f"{BASE_URL}/sources/ingest/web",
                json={"urls": parsed},
                timeout=120,
            )
        r.raise_for_status()
        d = r.json()
        return f"✅ Fetched **{d['documents_fetched']}** page(s) → **{d['chunks_ingested']}** chunk(s)"
    except Exception as e:
        return f"❌ {e}"


def github_ingest_fn(token, repos, branch):
    try:
        repo_list = [repo.strip() for repo in repos.split(",") if repo.strip()]
        with get_client() as client:
            r = client.post(
                f"{BASE_URL}/sources/ingest/github",
                json={
                    "token": token,
                    "repos": repo_list,
                    "branch": branch,
                },
                timeout=300,
            )
        r.raise_for_status()
        d = r.json()
        return f"✅ Fetched **{d['documents_fetched']}** file(s) → **{d['chunks_ingested']}** chunk(s)"
    except Exception as e:
        return f"❌ {e}"


def notion_ingest_fn(token, pages, dbs):
    try:
        pages_list = [p.strip() for p in pages.split(",") if p.strip()]
        dbs_list = [d.strip() for d in dbs.split(",") if d.strip()]

        with get_client() as client:
            r = client.post(
                f"{BASE_URL}/sources/ingest/notion",
                json={
                    "token": token,
                    "page_ids": pages_list,
                    "database_ids": dbs_list,
                },
                timeout=120,
            )

        r.raise_for_status()
        d = r.json()

        return f"✅ Fetched **{d['documents_fetched']}** page(s) → **{d['chunks_ingested']}** chunk(s)"

    except Exception as e:
        return f"❌ {e}"


def gdrive_ingest_fn(folder_id):
    try:
        with get_client() as client:
            r = client.post(
                f"{BASE_URL}/sources/ingest/gdrive",
                json={"folder_id": folder_id},
                timeout=300,
            )

        r.raise_for_status()
        d = r.json()

        return f"✅ Fetched **{d['documents_fetched']}** file(s) → **{d['chunks_ingested']}** chunk(s)"

    except Exception as e:
        return f"❌ {e}"


#  UI Builder 

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI RAG Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🤖 AI RAG Chat\n"
            "Ask questions about your uploaded documents. "
            "Supports streaming, session memory, and multiple LLM providers."
        )

        #  Chat Tab 
        with gr.Tab("💬 Chat"):
            with gr.Row(equal_height=True):
                provider_dd = gr.Dropdown(
                    choices=["ollama", "gemini", "groq", "openrouter", "openai"],
                    value="ollama",
                    label="Provider",
                    scale=1,
                )
                model_txt = gr.Textbox(
                    label="Model (blank = default)",
                    placeholder="e.g. gemini-2.0-flash-lite",
                    scale=2,
                )
                api_key_txt = gr.Textbox(
                    label="API Key (or set in .env)",
                    placeholder="Paste key — not saved",
                    type="password",
                    scale=2,
                )

            with gr.Row():
                system_instruction = gr.Textbox(
                    label="System Instruction (optional)",
                    placeholder="e.g. Answer only in bullet points.",
                    lines=2,
                    scale=5,
                )
                use_stream = gr.Checkbox(
                    label="⚡ Streaming",
                    value=True,
                    scale=1,
                    container=True,
                )

            chatbot = gr.Chatbot(
                height=460,
                type="messages",
                show_copy_button=True,
                avatar_images=(None, "🤖"),
                bubble_full_width=False,
            )

            with gr.Row():
                txt = gr.Textbox(
                    placeholder="Ask a question about your documents…",
                    scale=8,
                    show_label=False,
                    autofocus=True,
                )
                send_btn = gr.Button("Send ➤", scale=1, variant="primary")
                clear_btn = gr.Button("🗑 Clear", scale=1)

            chat_inputs = [txt, chatbot, system_instruction, provider_dd, model_txt, api_key_txt, use_stream]

            send_btn.click(
                chat_fn,
                inputs=chat_inputs,
                outputs=[chatbot, txt],
            )
            txt.submit(
                chat_fn,
                inputs=chat_inputs,
                outputs=[chatbot, txt],
            )
            clear_btn.click(lambda: ([], ""), None, [chatbot, txt])

        #  Upload Tab 
        with gr.Tab("📁 Upload"):
            gr.Markdown("Upload `.txt` or `.md` files to add them to the knowledge base.")
            file_input = gr.File(
                label="Select files",
                file_count="multiple",
            )
            upload_btn = gr.Button("⬆ Ingest Files", variant="primary")
            upload_status = gr.Markdown()
            upload_btn.click(
                upload_fn,
                inputs=[file_input],
                outputs=[upload_status],
                show_progress=True,
            )

        #  Sources Tab 
        with gr.Tab("🌐 Sources"):
            gr.Markdown("### Ingest from online sources")

            with gr.Tab("Web URLs"):
                web_urls = gr.Textbox(
                    label="URLs (one per line or comma-separated)",
                    lines=4,
                    placeholder="https://example.com/article\nhttps://docs.example.com",
                )
                web_btn = gr.Button("🌐 Fetch & Ingest", variant="primary")
                web_status = gr.Markdown()
                web_btn.click(web_ingest_fn, inputs=[web_urls], outputs=[web_status])

            with gr.Tab("GitHub"):
                gh_token = gr.Textbox(label="GitHub PAT Token", type="password")
                gh_repos = gr.Textbox(
                    label="Repos (comma-separated)",
                    placeholder="owner/repo1, owner/repo2",
                )
                gh_branch = gr.Textbox(
                    label="Branch (blank = default)",
                    placeholder="main",
                )
                gh_btn = gr.Button("🐙 Fetch & Ingest", variant="primary")
                gh_status = gr.Markdown()
                gh_btn.click(
                    github_ingest_fn,
                    inputs=[gh_token, gh_repos, gh_branch],
                    outputs=[gh_status],
                )

            with gr.Tab("Notion"):
                notion_token = gr.Textbox(label="Notion Integration Token", type="password")
                notion_pages = gr.Textbox(
                    label="Page IDs (comma-separated, or blank for all)",
                )
                notion_dbs = gr.Textbox(label="Database IDs (optional)")
                notion_btn = gr.Button("📝 Fetch & Ingest", variant="primary")
                notion_status = gr.Markdown()
                notion_btn.click(
                    notion_ingest_fn,
                    inputs=[notion_token, notion_pages, notion_dbs],
                    outputs=[notion_status],
                )

            with gr.Tab("Google Drive"):
                gr.Markdown(
                    "Place `credentials.json` in the project root.\n"
                    "First run will open a browser for Google OAuth consent."
                )
                gd_folder = gr.Textbox(
                    label="Folder ID (blank = all Drive files)",
                )
                gd_btn = gr.Button("🔗 Connect & Ingest", variant="primary")
                gd_status = gr.Markdown()
                gd_btn.click(gdrive_ingest_fn, inputs=[gd_folder], outputs=[gd_status])

        #  Manage Tab 
        with gr.Tab("🗂 Manage"):
            gr.Markdown("### Ingested Documents")
            refresh_btn = gr.Button("🔄 Refresh List")
            doc_list = gr.Markdown()

            gr.Markdown("### Preview Document")
            preview_dd = gr.Dropdown(
                label="Select document to preview",
                choices=[],
                interactive=True,
            )
            preview_btn = gr.Button("👁 Preview")
            preview_output = gr.Markdown()

            gr.Markdown("### Delete Document")
            delete_input = gr.Textbox(
                label="Document filename to delete",
                placeholder="e.g. knowledge.txt",
            )
            delete_btn = gr.Button("🗑 Delete", variant="stop")
            delete_status = gr.Markdown()

            # Refresh updates both doc list and preview dropdown
            refresh_btn.click(
                list_docs_fn,
                outputs=[doc_list, preview_dd],
            )
            preview_btn.click(
                preview_doc_fn,
                inputs=[preview_dd],
                outputs=[preview_output],
            )
            delete_btn.click(
                delete_doc_fn,
                inputs=[delete_input],
                outputs=[delete_status, delete_input],
            )
            # Auto-load on page open
            demo.load(list_docs_fn, outputs=[doc_list, preview_dd])

    return demo


#  Launch 

if __name__ == "__main__":
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=settings.ui_port,
        share=False,
    )