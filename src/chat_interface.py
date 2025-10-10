import gradio as gr
from gradio.themes import Base
from pathlib import Path
from langchain.schema import HumanMessage, Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.runnables import RunnableConfig
from graph import compile_graph
import os
from setup import _retriever


app = compile_graph()
checkpointer = app.checkpointer

def _to_text(x):
    # Transforma lista de paths a texto, si no rompe al llamar a OpenAI
    if isinstance(x, list):
        # Muestra solo nombres; evita paths largos
        return "\n".join(
            f"[archivo] {Path(p).name}" if isinstance(p, str) else str(p) for p in x
        )
    return "" if x is None else str(x)

def _to_history(message)-> dict[str, str]:
    return {"role": "user" if isinstance(message, HumanMessage) else "assistant", "content": str(message.content)}


def load_persisted_chat_history(config: RunnableConfig) -> list[dict[str, str]]:
    snap = app.get_state(config)                           
    msgs = snap.values.get("messages", [])
    return [_to_history(m) for m in msgs]

def chat_fn(message_dict, history, config):
    """
    Called by gr.ChatInterface.
    - message_dict: dict with 'text' and 'file'
    - history: list of [user, assistant], messages in the chat
    - config: RunnableConfig with 'thread_id'
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
    result = app.invoke({"messages": [HumanMessage(content=_to_text(user_message))]}, config)

    ai_messages = result.get("messages", [])
    answer = ""
    if ai_messages:
        answer = getattr(ai_messages[-1], "content", "") or str(ai_messages[-1])

    # Esto es para que se vea en la interfaz
    history = history + [(user_message, answer)]
    return answer


def create_chat_interface():
    # Load persisted chat history
    thread_id = "thread2"
    config = RunnableConfig({"configurable": {"thread_id": thread_id}})  # stable per chat/thread
    chat_history = load_persisted_chat_history(config)

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
                chatbot = gr.Chatbot(
                    chat_history,
                    type="messages",
                    height=720,
                    label=None,
                    show_label=False,
                    container=False,
                )
                gr.ChatInterface(
                    chat_fn,
                    multimodal=True,
                    type="messages",
                    chatbot=chatbot,
                    textbox=gr.MultimodalTextbox(
                        placeholder="Haz una consulta o sube un documento...",
                        sources=["microphone", "upload"],
                        file_types=[".pdf"],
                        file_count="multiple",
                        stop_btn=True,
                        elem_id="chat-input",
                    ),
                    additional_inputs=gr.State(
                        config
                    ),  # Los inputs adicionales a chat_fn tienen que ser gr.State u otro componente de gradio
                )

        # Sin esto los mensajes enviados desde el último inicio desparecen al recargar la página
        # hasta que se reinicia el servidor
        def _reload(config):
            return load_persisted_chat_history(config)

        demo.load(_reload, inputs=gr.State(config), outputs=chatbot)
    return demo
