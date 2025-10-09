import gradio as gr
from gradio.themes import Base
from pathlib import Path
from langchain.schema import AIMessage, HumanMessage, Document
from langchain_community.document_loaders import PyMuPDFLoader
from graph import compile_graph
import os
from setup import _retriever

_graph = compile_graph()


def _to_text(x):
    # Transforma lista de paths a texto, si no rompe al llamar a OpenAI
    if isinstance(x, list):
        # Muestra solo nombres; evita paths largos
        return "\n".join(
            f"[archivo] {Path(p).name}" if isinstance(p, str) else str(p) for p in x
        )
    return "" if x is None else str(x)


def _history_to_messages(history, user_msg):
    msgs = []
    for u, a in history:
        if u:
            msgs.append(HumanMessage(content=_to_text(u)))
        if a:
            msgs.append(AIMessage(content=_to_text(a)))
    msgs.append(HumanMessage(content=_to_text(user_msg)))
    return msgs


def chat_fn(message_dict, history):
    """
    Called by gr.ChatInterface.
    - message_dict: dict with 'text' and 'file'
    - history: list of [user, assistant]
    """
    # 1) turn chat into MessagesState
    user_message = message_dict["text"]
    files = message_dict.get("files", [])
    if files:
        respuestas = []
        for file_path in files:
            try:
                if file_path.lower().endswith(".pdf"):
                    # Reutilizar el retriever para cada PDF
                    loader = PyMuPDFLoader(file_path)
                    page_docs = loader.load()
                    full_text = "\n".join(d.page_content for d in page_docs)
                    docs = [
                        Document(
                            page_content=full_text,
                            metadata={"source": Path(file_path).name},
                        )
                    ]
                    _retriever.add_documents(docs)
                    respuestas.append(
                        f"Archivo: {os.path.basename(file_path)} procesado con éxito."
                    )
                else:
                    respuestas.append(
                        f"Tipo de archivo no soportado: {os.path.basename(file_path)}"
                    )
            except Exception as e:
                respuestas.append(
                    f"Error procesando {os.path.basename(file_path)}: {e}"
                )
    if not user_message.strip():
        return "\n\n".join(respuestas)
    msgs = _history_to_messages(history, user_message)
    result = _graph.invoke(
        {"messages": msgs}, config={"configurable": {"thread_id": "thread1"}}
    )
    ai_messages = result.get("messages", [])
    answer = ""
    if ai_messages:
        answer = getattr(ai_messages[-1], "content", "") or str(ai_messages[-1])
    return answer


def create_chat_interface():
    with gr.Blocks(
        title="Demo de RAG para Chatbot",
        theme=Base(),
        fill_height=True,
    ) as demo:
        with gr.Row():
            with gr.Sidebar():
                gr.Markdown("### Chats")
                gr.Button("Nuevo chat", variant="primary")
                gr.Markdown("---")
                gr.Button("Chat 1")
                gr.Button("Chat 2")
                gr.Button("Chat 3")

            with gr.Column():
                gr.ChatInterface(
                    chat_fn,
                    multimodal=True,
                    chatbot=gr.Chatbot(
                        height=720,
                        label=None,
                        show_label=False,
                        container=False,
                    ),
                    textbox=gr.MultimodalTextbox(
                        placeholder="Haz una consulta o sube un documento...",
                        sources=["microphone", "upload"],
                        file_types=[".pdf"],
                        file_count="multiple",
                        stop_btn=True,
                        elem_id="chat-input",
                    ),
                )
    return demo
