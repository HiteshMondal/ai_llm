import gradio as gr
import httpx
from app.utils.config import get_settings

settings = get_settings()
BASE_URL = f"http://{settings.app_host}:{settings.app_port}"


def chat_fn(question: str, history: list):
    if not question.strip():
        return history, ""

    try:
        resp = httpx.post(f"{BASE_URL}/chat", json={"question": question, "k": 4}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        answer = data["answer"]
        sources = data.get("sources", [])

        if sources:
            src_text = "\n\n**Sources:**\n" + "\n".join(
                f"- {s['metadata'].get('source', 'unknown')} (chunk excerpt: {s['content'][:80]}...)"
                for s in sources
            )
            answer += src_text
    except Exception as e:
        answer = f"❌ Error: {e}"

    history.append((question, answer))
    return history, ""


def upload_fn(file):
    if file is None:
        return "No file selected."
    try:
        with open(file.name, "rb") as f:
            resp = httpx.post(
                f"{BASE_URL}/ingest",
                files={"file": (file.name.split("/")[-1], f)},
                timeout=120,
            )
        resp.raise_for_status()
        data = resp.json()
        return f"✅ Ingested **{data['filename']}** — {data['chunks_ingested']} chunk(s) stored."
    except Exception as e:
        return f"❌ Upload failed: {e}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI RAG Chat") as demo:
        gr.Markdown("# 🤖 AI RAG Chat\nAsk questions about your uploaded documents.")

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Conversation", height=450)
            with gr.Row():
                txt = gr.Textbox(placeholder="Ask a question...", scale=8, show_label=False)
                send_btn = gr.Button("Send", scale=1)

            send_btn.click(chat_fn, inputs=[txt, chatbot], outputs=[chatbot, txt])
            txt.submit(chat_fn, inputs=[txt, chatbot], outputs=[chatbot, txt])

        with gr.Tab("Upload Documents"):
            file_input = gr.File(label="Upload a PDF, TXT, MD, or DOCX file")
            upload_btn = gr.Button("Ingest")
            upload_status = gr.Markdown()
            upload_btn.click(upload_fn, inputs=[file_input], outputs=[upload_status])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)