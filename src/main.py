import gradio as gr
import pandas as pd
import time
import os
import shutil
from pathlib import Path
from langchain.schema import Document
from langchain.schema import HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.runnables import RunnableConfig
from db import (
    get_user_id,
    get_chat_by_id,
    list_chats,
    create_chat,
    touch_chat,
    add_file,
    get_chat_id_by_thread,
    get_files_by_chat_id,
    delete_chat_by_thread,
    persist_message,
    load_chat_messages,
)
from auth import verify
from graph import graph, retriever
from gradio import ChatMessage
from stytle import gemis_theme, custom_css


def _to_history(message) -> ChatMessage | None:
    if message["type"] == "file":
        role = "user"
        content = gr.FileData(path=message["content"])
    elif message is not None:
        role = message["role"]
        content = message["content"]
    else:
        return None
    return ChatMessage(role=role, content=content)


def _reload(user: dict):
    if user is None or user.get("username") is None or user.get("ttl", 0) < time.time():
        return gr.update(), None, None, gr.update(choices=[], value=None)
    username = user.get("username")
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
        choices = sorted(choices, key=lambda x: x[0])
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
    new_choices = sorted(new_choices, key=lambda x: x[0])
    return [], tid, gr.update(choices=new_choices, value=tid)


def _delete_chat(user_id, thread_id):
    delete_chat_by_thread(thread_id)
    remaining_threads = list_chats(user_id)
    if len(remaining_threads) > 0:
        new_tid = remaining_threads[0].get("thread_id")
        new_choices = [
            (f"{t.get('title', '')}", t.get("thread_id")) for t in remaining_threads
        ]
        return (
            [],
            new_tid,
            gr.update(choices=sorted(new_choices, key=lambda x: x[0]), value=new_tid),
        )
    else:
        return _new_chat(user_id)


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
    if not config["configurable"].get("thread_id"):
        return []
    thread_id = config["configurable"].get("thread_id")
    chat_id = get_chat_id_by_thread(thread_id)
    persisted = load_chat_messages(chat_id)
    return [h for h in (_to_history(m) for m in persisted) if h is not None]


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
            persist_message(
                content=path,
                role="user",
                type="file",
                chat_id=get_chat_id_by_thread(thread_id),
                thread_id=thread_id,
            )
            loader = PyMuPDFLoader(path)
            page_docs = loader.load()
            save_path = os.path.join(
                f"/home/moya/Langchain-chatbot/test_data/{user_id}/thread_{thread_id}",
                Path(path).name,
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            shutil.copy2(path, save_path)
            # Example: Saving content to the file
            chat_id = get_chat_id_by_thread(thread_id)
            file_id = add_file(
                user_id,
                chat_id,
                original_name=Path(path).name,
                stored_path=Path(save_path),
            )
            if file_id:
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
                retriever.add_documents(docs)
                uploaded_files.append(Path(path).name)
    if uploaded_files and user_message.strip() == "":
        history = history + [
            {
                "role": "assistant",
                "content": "Tus archivos han sido subidos correctamente. ¿En qué puedo ayudarte con ellos?",
            }
        ]
        return history, get_files(thread_id)
    result = graph.invoke({"messages": [HumanMessage(content=user_message)]}, config)
    ai_messages = result.get("messages", [])
    answer = ""
    if ai_messages:
        answer = getattr(ai_messages[-1], "content", "") or str(ai_messages[-1])
    history = history + [{"role": "assistant", "content": answer}]
    persist_message(
        content=user_message,
        role="user",
        type="text",
        chat_id=get_chat_id_by_thread(thread_id),
        thread_id=thread_id,
    )
    persist_message(
        content=answer,
        role="assistant",
        type="text",
        chat_id=get_chat_id_by_thread(thread_id),
        thread_id=thread_id,
    )
    return history, get_files(thread_id)


def do_login(user: str, password: str):
    ok, _ = verify(user, password)
    if not ok:
        # keep login visible, show error
        return (
            gr.update(),
            gr.update(),
            gr.Markdown(
                "Usuario o contraseña inválidos.",
                elem_id="login-message",
                elem_classes=["error"],
            ),
            gr.update(),
        )
    return (
        gr.update(visible=False),  # hide login
        gr.update(visible=True),  # show app
        gr.Markdown(
            "",
            elem_id="login-message",
            elem_classes=["success"],
        ),
        {"username": user, "ttl": time.time() + 3600},  # set username state
    )


def do_logout():
    return (gr.update(visible=True), gr.update(visible=False), "", False)


def restore_session(stored_user: dict):
    if (
        stored_user is None
        or stored_user.get("username") is None
        or stored_user.get("ttl", 0) < time.time()
    ):
        return (
            gr.update(visible=True),  # login panel visible
            gr.update(visible=False),  # app hidden
            "",
            {"username": None, "ttl": 0},
        )
    return (
        gr.update(visible=False),  # hide login
        gr.update(visible=True),  # show app
        f"Hola, **{stored_user.get('username')}** 👋",
        gr.update(),
    )



with gr.Blocks(title="Demo de Chatbot", theme=gemis_theme, css=custom_css) as demo:
    auth = gr.BrowserState(
        {"username": None, "ttl": 0},
    )
    with gr.Row(elem_id="login-wrapper", visible=True) as login_column:
        with gr.Column(elem_id="login-logo-container"):
            gr.Image(
                value="/home/moya/Langchain-chatbot/assets/gemis-logo.png",
                width=280,
                show_label=False,
                show_download_button=False,
                show_share_button=False,
                show_fullscreen_button=False,
                container=False,
                elem_id="login-logo",
            )
        with gr.Column(elem_id="login-card", visible=True):
            gr.Markdown("# Iniciar sesión")
            gr.HTML("<p class='subtitle'>Ingresá tus credenciales para continuar</p>")

            gr.Markdown("Nombre de usuario", label="input-label")
            user = gr.Textbox(
                elem_id="login-username",
                label="Nombre de usuario",
                value="",
                type="text",
                max_lines=1,
                container=False,
                elem_classes=["username-input"],
            )

            gr.Markdown("Contraseña", label="input-label")
            pwd = gr.Textbox(
                elem_id="login-password",
                label="Contraseña",
                type="password",
                lines=1,
                container=False,
            )

            login_btn = gr.Button("Ingresar", elem_id="login-btn", variant="primary")
            login_msg = gr.Markdown("", elem_id="login-message")

    with gr.Blocks(fill_height=True):
        user_id = gr.State(None)
        thread_id = gr.State(None)
        message = gr.State(None)

        with gr.Column(visible=False) as sidebar_and_chat:
            with gr.Sidebar():
                gr.Markdown("### Tus Chats")
                chat_selector = gr.Dropdown(
                    label="Seleccionar chat",
                    choices=[],
                    value=None,
                    container=False,
                    interactive=True,
                )
                new_btn = gr.Button("Nuevo chat", variant="primary")
                del_btn = gr.Button("Eliminar chat", variant="stop")
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
                    group_consecutive_messages=True,
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
                    lambda: gr.MultimodalTextbox(interactive=True),
                    None,
                    [multimodal],
                )
            new_btn.click(
                _new_chat,
                inputs=[user_id],
                outputs=[chatbot, thread_id, chat_selector],
            ).then(
                lambda: gr.MultimodalTextbox(interactive=True),
                None,
                [multimodal],
            )

            del_btn.click(
                _delete_chat,
                inputs=[user_id, thread_id],
                outputs=[chatbot, thread_id, chat_selector],
            ).then(
                lambda: gr.MultimodalTextbox(interactive=True),
                None,
                [multimodal],
            )

            chat_selector.change(
                _switch_chat,
                inputs=[chat_selector, user_id],
                outputs=[chatbot, thread_id, files_table],
            ).then(
                lambda: gr.MultimodalTextbox(interactive=True),
                None,
                [multimodal],
            )

            gr.on(
                triggers=[login_btn.click, user.submit, pwd.submit],
                fn=do_login,
                inputs=[user, pwd],
                outputs=[login_column, sidebar_and_chat, login_msg, auth],
            ).success(
                _reload,
                inputs=auth,
                outputs=[chatbot, thread_id, user_id, chat_selector],
            )

        user.change(
            lambda _: gr.Markdown("", elem_id="login-message", elem_classes=[]),
            inputs=[user],
            outputs=[login_msg],
        )
        pwd.change(
            lambda _: gr.Markdown("", elem_id="login-message", elem_classes=[]),
            inputs=[pwd],
            outputs=[login_msg],
        )
        demo.load(
            restore_session,
            inputs=[auth],
            outputs=[login_column, sidebar_and_chat, login_msg, auth],
        ).success(
            _reload,
            inputs=auth,
            outputs=[chatbot, thread_id, user_id, chat_selector],
        )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=5).launch()
