import gradio as gr
from dotenv import load_dotenv

from pro_implementation.answer import answer_question

load_dotenv(override=True)


def chat(history):
    last_message = history[-1]["content"]
    prior = history[:-1]
    answer, _ = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(
        primary_hue="blue",
        font=["Inter", "system-ui", "sans-serif"]
    )

    with gr.Blocks(title="BONYAD Assistant", theme=theme) as ui:
        gr.Markdown(
            """
            # 🏗️ BONYAD Assistant
            Your expert guide for the BONYAD construction management platform.
            Ask me anything about how to use the app!
            """
        )

        chatbot = gr.Chatbot(
            label="💬 Chat",
            height=600,
            type="messages",
            show_copy_button=True,
            avatar_images=(None, "🏗️")
        )
        message = gr.Textbox(
            placeholder="Ask about projects, contractors, invoices, reports...",
            show_label=False,
            container=False
        )

        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot]
        ).then(
            chat,
            inputs=chatbot,
            outputs=chatbot
        )

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()