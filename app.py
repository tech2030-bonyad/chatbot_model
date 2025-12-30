import gradio as gr
from dotenv import load_dotenv

from pro_implementation.answer import answer_question

load_dotenv(override=True)


def chat(history):
    try:
        if not history or len(history) == 0:
            return history
        
        # Get the last message from history
        last_message_dict = history[-1]
        if isinstance(last_message_dict, dict):
            last_message = last_message_dict.get("content", "")
        else:
            # Handle case where history might be in a different format
            last_message = str(last_message_dict)
        
        if not last_message or not last_message.strip():
            return history
        
        # Get prior messages, filtering to ensure they're in the right format
        prior = []
        for msg in history[:-1]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                prior.append(msg)
        
        answer, _ = answer_question(last_message, prior)
        history.append({"role": "assistant", "content": answer})
        return history
    except Exception as e:
        error_message = f"Error: {str(e)}"
        import traceback
        traceback.print_exc()
        if history:
            history.append({"role": "assistant", "content": error_message})
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
            outputs=chatbot,
            show_progress=True
        )

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()