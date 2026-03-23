import gradio as gr
import httpx

from app.config import get_settings


settings = get_settings()

BASE_URL = f"http://127.0.0.1:{settings.app_port}"

client = httpx.Client(timeout=60)

# Chat handler

def chat_fn(question: str, history: list[dict] | None):

    history = history or []

    if not question.strip():
        return history, ""

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "⏳ Thinking..."})

    try:

        resp = client.post(
            f"{BASE_URL}/chat",
            json={"question": question, "k": 4},
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

    except Exception as e:

        answer = f"❌ Error contacting backend:\n{e}"

    # replace placeholder
    history[-1]["content"] = answer

    return history, ""


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

def build_ui() -> gr.Blocks:

    with gr.Blocks(title="AI RAG Chat") as demo:

        gr.Markdown(
            "# 🤖 AI RAG Chat\n"
            "Ask questions about your uploaded documents."
        )

        with gr.Tab("Chat"):

            chatbot = gr.Chatbot(height=450, type="messages", allow_tags=False)
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
                inputs=[txt, chatbot],
                outputs=[chatbot, txt],
            )

            txt.submit(
                chat_fn,
                inputs=[txt, chatbot],
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

            status = gr.Markdown()

            upload_btn.click(
                upload_fn,
                inputs=[file_input],
                outputs=[status],
                show_progress=True,
            )

    return demo


# Launch UI

if __name__ == "__main__":

    build_ui().launch(
        server_name="0.0.0.0",
        server_port=settings.ui_port,
        share=False,
    )