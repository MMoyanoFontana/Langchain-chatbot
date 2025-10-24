import gradio as gr
from gradio.themes import Base
import pandas as pd
from pathlib import Path
from langchain.schema import HumanMessage, Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from db import (
    get_user_id,
    get_chat_by_id,
    list_chats,
    create_chat,
    touch_chat,
    add_file,
    get_chat_id_by_thread,
    get_sha,
    get_files_by_chat_id,
)
from graph import compile_graph
from setup import _retriever

app = compile_graph()
checkpointer = app.checkpointer


def _to_history(message) -> dict[str, str] | None:
    if isinstance(message, HumanMessage):
        role = "user"
        text = message.content
    elif isinstance(message, AIMessage):
        role = "assistant"
        text = getattr(message, "content", "")
    else:
        return None
    return {"role": role, "content": text}


def _reload(req: gr.Request):
    username = req.username
    uid = get_user_id(username)
    placeholder = f"Hola, {username.capitalize()} ¿En qué puedo ayudarte hoy?"
    # último hilo del usuario o crea uno si no hay
    threads = list_chats(uid)
    if len(threads) == 0:
        chat_id = create_chat(uid, "Chat 1")
        tid = get_chat_by_id(chat_id).get("thread_id")
        choices = [("Chat 1", tid)]
    else:
        tid = threads[0].get("thread_id")
        choices = [(f"{t.get('title', '')}", t.get("thread_id")) for t in threads]
    cfg = RunnableConfig({"configurable": {"thread_id": tid, "user_id": uid}})
    hist = load_persisted_chat_history(cfg)
    return (
        gr.update(value=hist, placeholder=placeholder),
        tid,
        uid,
        gr.update(choices=choices, value=tid),
    )


def _new_chat(user_id):
    threads = list_chats(user_id)
    dd_choices = [(f"{t.get('title', '')}", t.get("thread_id")) for t in threads]
    idx = 1
    base = "Chat"
    while f"{base} {idx}" in [c[0] for c in dd_choices]:
        idx += 1
    title = f"{base} {idx}"
    id = create_chat(int(user_id), title)
    tid = get_chat_by_id(id).get("thread_id")
    new_choices = [(title, tid)] + dd_choices
    return [], tid, gr.update(choices=new_choices, value=tid)


def get_files(tid):
    chat_id = get_chat_id_by_thread(tid)
    archivos = get_files_by_chat_id(chat_id)
    if len(archivos) > 0:
        return pd.DataFrame([archivo["original_name"] for archivo in archivos])
    return pd.DataFrame([])


def _switch_chat(tid, user_id):
    cfg = RunnableConfig({"configurable": {"thread_id": tid, "user_id": user_id}})
    hist = load_persisted_chat_history(cfg)
    archivos = get_files(tid)
    return hist, tid, archivos


def load_persisted_chat_history(config: RunnableConfig) -> list[dict[str, str]]:
    snap = app.get_state(config)
    msgs = snap.values.get("messages", [])
    return [h for h in (_to_history(m) for m in msgs) if h is not None]


def add_message(history, message):
    for x in message["files"]:
        history.append({"role": "user", "content": gr.File(value=x)})
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False), message


def bot(history, message, thread_id, user_id):
    touch_chat(thread_id)
    config = RunnableConfig(
        {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    )
    user_message = message["text"]
    uploaded_files = []
    for path in message["files"]:
        if path.lower().endswith(".pdf"):
            loader = PyMuPDFLoader(path)
            page_docs = loader.load()
            chat_id = get_chat_id_by_thread(thread_id)
            file_id = add_file(
                user_id, chat_id, original_name=Path(path).name, stored_path=Path(path)
            )
            if file_id:
                file_sha = get_sha(file_id)
                full_text = "\n".join(d.page_content for d in page_docs)
                docs = [
                    Document(
                        page_content=full_text,
                        metadata={
                            "source": Path(path).name,
                            "user_id": user_id,
                            "thread_id": thread_id,
                        },
                    )
                ]
                _retriever.add_documents(docs, id=file_sha)
                uploaded_files.append(Path(path).name)
    preface = []
    if uploaded_files:
        preface.append(
            SystemMessage(
                content=(
                    "Se subieron nuevos archivos a este chat: "
                    + ", ".join(uploaded_files)
                    + ". Cada página fue indexada con metadata {source, user_id, thread_id}. "
                    "Si la consulta puede relacionarse con estos documentos, USA la herramienta `retrieve` "
                    "con el query del usuario para recuperar pasajes relevantes."
                    "Si el usuario solo sube archivos sin hacer una pregunta relacionada, responde que se procesaron con éxito."
                )
            )
        )
    result = app.invoke(
        {"messages": preface + [HumanMessage(content=user_message)]}, config
    )
    ai_messages = result.get("messages", [])
    answer = ""
    if ai_messages:
        answer = getattr(ai_messages[-1], "content", "") or str(ai_messages[-1])
    history = history + [{"role": "assistant", "content": answer}]
    return history, get_files(thread_id)


def create_chat_interface():
    with gr.Blocks(
        title="Demo de Chatbot",
        theme=Base(),
        fill_height=True,
    ) as demo:
        user_id = gr.State(value=1)
        thread_id = gr.State(value="thread1")
        message = gr.State(value=None)

        with gr.Row():
            with gr.Sidebar():
                gr.Markdown("### Tus Chats")
                chat_selector = gr.Dropdown(
                    label="Seleccionar chat", choices=[], value=None, interactive=True
                )
                new_btn = gr.Button("Nuevo chat", variant="primary")
                gr.Markdown("---")
                gr.Markdown("### Archivos disponibles en este chat")
                files_table = gr.Dataframe(
                    headers=None,
                    datatype=["str"],
                    interactive=False,
                    row_count=(0, "dynamic"),
                    col_count=(1, "fixed"),
                    wrap=True,
                    label=None,
                )

            with gr.Column():
                chatbot = gr.Chatbot(
                    type="messages",
                    height=720,
                    latex_delimiters=[
                        {"left": "$", "right": "$", "display": False},
                        {"left": "$$", "right": "$$", "display": True},
                    ],
                    label=None,
                    show_label=False,
                    container=False,
                )
                multimodal = gr.MultimodalTextbox(
                    interactive=True,
                    file_count="multiple",
                    placeholder="Escribí tu mensaje o subí archivos.",
                    show_label=False,
                    sources=["microphone", "upload"],
                )

                chat_msg = multimodal.submit(
                    add_message,
                    show_progress="hidden",
                    inputs=[chatbot, multimodal],
                    outputs=[chatbot, multimodal, message],
                )
                bot_msg = chat_msg.then(
                    bot,
                    inputs=[chatbot, message, thread_id, user_id],
                    outputs=[chatbot, files_table],
                )
                bot_msg.then(
                    lambda: gr.MultimodalTextbox(interactive=True), None, [multimodal]
                )

        demo.load(
            _reload,
            inputs=None,
            outputs=[chatbot, thread_id, user_id, chat_selector],
        )
        new_btn.click(
            _new_chat,
            inputs=[user_id],
            outputs=[chatbot, thread_id, chat_selector],
        ).then(lambda: gr.MultimodalTextbox(interactive=True), None, [multimodal])

        chat_selector.change(
            _switch_chat,
            inputs=[chat_selector, user_id],
            outputs=[chatbot, thread_id, files_table],
        ).then(lambda: gr.MultimodalTextbox(interactive=True), None, [multimodal])

    return demo
