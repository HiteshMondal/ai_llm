import gradio as gr
import httpx

from app.config import get_settings


settings = get_settings()

BASE_URL = f"http://127.0.0.1:{settings.app_port}"

client = httpx.Client(timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10))


# Upload handler

def upload_fn(files):

    if not files or len(files) == 0:
        return "No file selected."

    results = []

    for file in files:

        try:

            with open(file.name, "rb") as f:

                resp = client.post(
                    f"{BASE_URL}/ingest",
                    files={"file": (file.name.split("/")[-1], f)},
                    timeout=120,
                )

            resp.raise_for_status()

            data = resp.json()

            results.append(
                f"✅ Ingested **{data['filename']}** "
                f"— {data['chunks_ingested']} chunk(s)."
            )

        except Exception as e:

            results.append(
                f"❌ Upload failed for **{file.name}**:\n{e}"
            )

    return "\n\n".join(results)

# UI Layout

def list_docs_fn():
    try:
        resp = client.get(f"{BASE_URL}/documents")
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
        if not docs:
            return "No documents ingested yet."
        return "\n".join(
            f"- **{d['source']}** — {d['chunk_count']} chunk(s)"
            for d in docs
        )
    except Exception as e:
        return f"❌ Error: {e}"

def delete_doc_fn(source: str):
    source = source.strip()
    if not source:
        return "Enter a document name to delete.", gr.update()
    try:
        from urllib.parse import quote
        resp = client.delete(f"{BASE_URL}/documents/{quote(source)}")
        resp.raise_for_status()
        data = resp.json()
        return (
            f"✅ Deleted **{data['source']}** — {data['deleted_chunks']} chunk(s) removed.",
            gr.update(value="")
        )
    except Exception as e:
        return f"❌ Error: {e}", gr.update()

def chat_fn(question: str, history: list[dict] | None, system_instruction: str):
    history = history or []
    if not question.strip():
        return history, ""
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "⏳ Thinking..."})
    try:
        resp = client.post(
            f"{BASE_URL}/chat",
            json={
                "question": question,
                "k": 4,
                "system_instruction": system_instruction,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["answer"]
        sources = data.get("sources", [])
        if sources:
            answer += "\n\n**Sources:**\n"
            answer += "\n".join(
                f"- {s['metadata'].get('source', '?')} "
                f"— {s['content'][:80]}..."
                for s in sources
            )
    except httpx.TimeoutException:
        answer = "⏳ Request timed out. The model is taking too long — try a shorter question or wait and retry."
    except Exception as e:
        answer = f"❌ Error contacting backend:\n{e}"
    history[-1]["content"] = answer
    return history, ""

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI RAG Chat") as demo:
        gr.Markdown(
            "# 🤖 AI RAG Chat\n"
            "Ask questions about your uploaded documents."
        )
        with gr.Tab("Chat"):
            system_instruction = gr.Textbox(
                label="System Instruction (optional)",
                placeholder="e.g. Answer only in bullet points. Be concise.",
                lines=2,
            )
            chatbot = gr.Chatbot(height=400, type="messages", allow_tags=False)
            clear_btn = gr.Button("Clear Chat")
            with gr.Row():
                txt = gr.Textbox(
                    placeholder="Ask a question...",
                    scale=8,
                    show_label=False,
                )
                send_btn = gr.Button("Send", scale=1)
            send_btn.click(
                chat_fn,
                inputs=[txt, chatbot, system_instruction],
                outputs=[chatbot, txt],
            )
            txt.submit(
                chat_fn,
                inputs=[txt, chatbot, system_instruction],
                outputs=[chatbot, txt],
            )
            clear_btn.click(
                lambda: ([], ""),
                None,
                [chatbot, txt],
            )

        with gr.Tab("Upload"):
            file_input = gr.File(
                label="Upload TXT or MD files",
                file_count="multiple"
            )
            upload_btn = gr.Button("Ingest")
            upload_status = gr.Markdown()
            upload_btn.click(
                upload_fn,
                inputs=[file_input],
                outputs=[upload_status],
                show_progress=True,
            )

        with gr.Tab("Manage"):
            gr.Markdown("### Ingested Documents")
            refresh_btn = gr.Button("🔄 Refresh List")
            doc_list = gr.Markdown()
            gr.Markdown("### Delete Document")
            delete_input = gr.Textbox(
                label="Document name to delete",
                placeholder="e.g. Ai_data.txt",
            )
            delete_btn = gr.Button("🗑️ Delete", variant="stop")
            delete_status = gr.Markdown()
            refresh_btn.click(list_docs_fn, outputs=[doc_list])
            delete_btn.click(
                delete_doc_fn,
                inputs=[delete_input],
                outputs=[delete_status, delete_input],
            )
            demo.load(list_docs_fn, outputs=[doc_list])

    return demo


# Launch UI

if __name__ == "__main__":

    build_ui().launch(
        server_name="0.0.0.0",
        server_port=settings.ui_port,
        share=False,
    )