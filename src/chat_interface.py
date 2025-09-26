import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage


def history_to_messages(history, user_msg):
    """Convert chat history to MessagesState format."""
    msgs = []
    for u, a in history:
        if u:
            msgs.append(HumanMessage(content=u))
        if a:
            msgs.append(AIMessage(content=a))
    msgs.append(HumanMessage(content=user_msg))
    return msgs


def create_chat_interface(graph):
    """Create a Gradio chat interface with a sidebar for conversations 
    and a button to create new ones."""
    conversations = {"Chat 1": []}
    selected_conversation = "Default"

    def chat_fn(user_message, history):
        msgs = history_to_messages(history, user_message)
        result = graph.invoke(
            {"messages": msgs},
            config={"configurable": {"thread_id": "thread1"}},
        )
        ai_messages = result.get("messages", [])
        answer = ""
        if ai_messages:
            answer = getattr(ai_messages[-1], "content", "") or str(ai_messages[-1])
        return answer

    def create_new_conversation():
        new_name = f"Chat {len(conversations) + 1}"
        conversations[new_name] = []
        return list(conversations.keys()), new_name, conversations[new_name]

    with gr.Blocks(
        title="Chat Demo with Sidebar Conversations", theme=gr.themes.Soft()
    ) as demo:
        with gr.Sidebar():
            gr.Markdown("## Conversaciones")
            conversation_list = gr.Radio(
                choices=list(conversations.keys()),
                value="Default",
            )
            gr.Button("Nueva conversación").click(
                fn=create_new_conversation,
                inputs=[],
                outputs=[conversation_list],
            )
        with gr.Row():
            chatbot = gr.Chatbot(height=640, show_copy_button=True)
            gr.ChatInterface(
                fn=chat_fn,
                chatbot=chatbot,
                textbox=gr.Textbox(placeholder="Pregunta sobre el PDF…"),
            )
    return demo
