import gradio as gr
import time
import os
import shutil
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
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
    rename_chat,
)
from auth import verify
from title_setter import _generate_title_openai
from graph import graph, retriever
from gradio import ChatMessage
from style import gemis_theme, custom_css

# Constants
MAX_FILE_SIZE_MB = 10
ALLOWED_FILE_TYPES = [".pdf"]
SESSION_TIMEOUT = 3600  # 1 hour


def _to_history(message) -> ChatMessage | None:
    """Convert database message to ChatMessage format."""
    try:
        if message["type"] == "file":
            return ChatMessage(
                role="user", content=gr.FileData(path=message["content"])
            )
        elif message is not None:
            return ChatMessage(role=message["role"], content=message["content"])
    except Exception as e:
        print(f"Error converting message to history: {e}")
    return None


def _reload_chats(uid: int):
    """Reload user's chat list."""
    threads = list_chats(uid)
    if len(threads) == 0:
        chat_id = create_chat(uid, "Chat 1")
        tid = get_chat_by_id(chat_id).get("thread_id")
        choices = [("Chat 1", tid)]
    else:
        tid = threads[0].get("thread_id")
        choices = [
            (f"{t.get('title', 'Sin título')}", t.get("thread_id")) for t in threads
        ]
        choices = choices
    return gr.update(choices=choices, value=tid)


def _reload(user: dict):
    """Reload user session and chat history."""
    if user is None or user.get("username") is None or user.get("ttl", 0) < time.time():
        return gr.update(), None, None, gr.update(choices=[], value=None)

    try:
        username = user.get("username")
        uid = get_user_id(username)
        placeholder = f"Hola, {username.capitalize()} ¿En qué puedo ayudarte hoy?"

        threads = list_chats(uid)
        if len(threads) == 0:
            chat_id = create_chat(uid, "Chat 1")
            tid = get_chat_by_id(chat_id).get("thread_id")
            choices = [("Chat 1", tid)]
        else:
            tid = threads[0].get("thread_id")
            choices = [
                (f"{t.get('title', 'Sin título')}", t.get("thread_id")) for t in threads
            ]
            choices = choices

        cfg = RunnableConfig({"configurable": {"thread_id": tid, "user_id": uid}})
        hist = load_persisted_chat_history(cfg)

        return (
            gr.update(value=hist, placeholder=placeholder),
            tid,
            uid,
            gr.update(choices=choices, value=tid),
        )
    except Exception as e:
        print(f"Error reloading session: {e}")
        return gr.update(), None, None, gr.update(choices=[], value=None)


def _new_chat(user_id):
    """Create a new chat thread."""
    try:
        threads = list_chats(user_id)
        dd_choices = [(f"{t.get('title', '')}", t.get("thread_id")) for t in threads]

        # Find next available chat number
        idx = 1
        base = "Chat"
        while f"{base} {idx}" in [c[0] for c in dd_choices]:
            idx += 1

        title = f"{base} {idx}"
        id = create_chat(int(user_id), title)
        tid = get_chat_by_id(id).get("thread_id")

        new_choices = [(title, tid)] + dd_choices

        return (
            [],
            tid,
            gr.update(choices=new_choices, value=tid),
            gr.update(value=[]),
        )
    except Exception as e:
        print(f"Error creating new chat: {e}")
        gr.Warning("No se pudo crear el nuevo chat. Por favor, intenta nuevamente.")
        return gr.update(), gr.update(), gr.update(), gr.update()


def _delete_chat(user_id, thread_id):
    """Delete current chat with error handling."""
    try:
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
                gr.update(choices=new_choices, value=new_tid),
                gr.update(value=[]),
            )
        else:
            return _new_chat(user_id)
    except Exception as e:
        print(f"Error deleting chat: {e}")
        gr.Warning("No se pudo eliminar el chat. Por favor, intenta nuevamente.")
        return gr.update(), gr.update(), gr.update(), gr.update()


def get_files(tid):
    """Get files for current chat with better formatting."""
    try:
        chat_id = get_chat_id_by_thread(tid)
        archivos = get_files_by_chat_id(chat_id)
        # Devolvemos rutas absolutas/validas en disco
        paths = []
        for a in archivos:
            # a["stored_path"] ya debería ser una Path o string guardada en DB
            p = str(a["stored_path"])
            if os.path.exists(p):
                paths.append(p)
        return paths
    except Exception as e:
        print(f"Error getting files: {e}")
        return []


def _switch_chat(tid, user_id):
    """Switch to different chat thread."""
    try:
        cfg = RunnableConfig({"configurable": {"thread_id": tid, "user_id": user_id}})
        hist = load_persisted_chat_history(cfg)
        archivos = get_files(tid)
        return hist, tid, archivos
    except Exception as e:
        print(f"Error switching chat: {e}")
        gr.Warning("No se pudo cargar el chat. Por favor, intenta nuevamente.")
        return gr.update(), gr.update(), gr.update()


def load_persisted_chat_history(config: RunnableConfig) -> list[dict[str, str]]:
    """Load chat history from database."""
    try:
        if not config["configurable"].get("thread_id"):
            return []

        thread_id = config["configurable"].get("thread_id")
        chat_id = get_chat_id_by_thread(thread_id)
        persisted = load_chat_messages(chat_id)

        return [h for h in (_to_history(m) for m in persisted) if h is not None]
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []


def validate_file(file_path: str) -> tuple[bool, str]:
    """Validate uploaded file."""
    # Check file extension
    if not any(file_path.lower().endswith(ext) for ext in ALLOWED_FILE_TYPES):
        return (
            False,
            f"Tipo de archivo no permitido. Solo se aceptan: {', '.join(ALLOWED_FILE_TYPES)}",
        )

    # Check file size
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return (
                False,
                f"Archivo demasiado grande. Tamaño máximo: {MAX_FILE_SIZE_MB}MB",
            )
    except Exception as e:
        return False, f"Error al validar archivo: {str(e)}"

    return True, ""


def add_message(history, message):
    """Add user message to chat history."""
    try:
        for x in message["files"]:
            history.append({"role": "user", "content": gr.File(value=x)})
        if message["text"] is not None and message["text"].strip():
            history.append({"role": "user", "content": message["text"]})
        return history, gr.MultimodalTextbox(value=None, interactive=False), message
    except Exception as e:
        print(f"Error adding message: {e}")
        gr.Warning("Error al procesar el mensaje.")
        return history, gr.MultimodalTextbox(value=None, interactive=True), message


def bot(history, message, thread_id, user_id):
    """Process user message and generate bot response."""
    try:
        touch_chat(thread_id)
        config = RunnableConfig(
            {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        )
        user_message = message["text"] or ""
        uploaded_files = []

        # Process uploaded files
        for path in message["files"]:
            # Validate file
            is_valid, error_msg = validate_file(path)
            if not is_valid:
                gr.Warning(error_msg)
                continue

            if path.lower().endswith(".pdf"):
                try:
                    # Persist file message
                    persist_message(
                        content=path,
                        role="user",
                        type="file",
                        chat_id=get_chat_id_by_thread(thread_id),
                        thread_id=thread_id,
                    )

                    # Load PDF content
                    loader = PyMuPDFLoader(path)
                    page_docs = loader.load()

                    # Sanitize and save file
                    safe_filename = Path(path).name
                    save_dir = f"/home/moya/Langchain-chatbot/test_data/{user_id}/thread_{thread_id}"
                    save_path = os.path.join(save_dir, safe_filename)

                    os.makedirs(save_dir, exist_ok=True)
                    shutil.copy2(path, save_path)

                    # Add to database
                    chat_id = get_chat_id_by_thread(thread_id)
                    file_id = add_file(
                        user_id,
                        chat_id,
                        original_name=safe_filename,
                        stored_path=Path(save_path),
                    )

                    if file_id:
                        # Add to vector store
                        full_text = "\n".join(d.page_content for d in page_docs)
                        docs = [
                            Document(
                                page_content=full_text,
                                metadata={
                                    "source": safe_filename,
                                    "user_id": user_id,
                                    "thread_id": thread_id,
                                },
                            )
                        ]
                        retriever.add_documents(docs)
                        uploaded_files.append(safe_filename)

                except Exception as e:
                    print(f"Error processing PDF {path}: {e}")
                    gr.Warning(f"Error al procesar {Path(path).name}")

        should_generate_title = (len(history) == 1 or len(history) == 2) and (
            user_message.strip() != "" or uploaded_files
        )
        # Handle file-only uploads
        if uploaded_files and user_message.strip() == "":
            if should_generate_title:
                title = _generate_title_openai(f"files: {uploaded_files}")
                rename_chat(thread_id, title)
            response = "Tus archivos han sido subidos correctamente. ¿En qué puedo ayudarte con ellos?"
            history = history + [{"role": "assistant", "content": response}]
            persist_message(
                content=response,
                role="assistant",
                type="text",
                chat_id=get_chat_id_by_thread(thread_id),
                thread_id=thread_id,
            )
            return history, get_files(thread_id)

        # Get bot response
        if user_message.strip():
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_message)]}, config
            )

            ai_messages = result.get("messages", [])
            answer = ""
            if ai_messages:
                answer = getattr(ai_messages[-1], "content", "") or str(ai_messages[-1])

            history = history + [{"role": "assistant", "content": answer}]

            # Persist messages
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

            if should_generate_title:
                title = _generate_title_openai(user_message)
                rename_chat(thread_id, title)

        return history, get_files(thread_id)

    except Exception as e:
        print(f"Error in bot function: {e}")
        error_msg = "Lo siento, ocurrió un error al procesar tu mensaje. Por favor, intenta nuevamente."
        history = history + [{"role": "assistant", "content": error_msg}]
        return history, get_files(thread_id)


def do_login(user: str, password: str):
    """Handle user login."""
    if not user or not password:
        return (
            gr.update(),
            gr.update(),
            gr.Markdown(
                "Por favor, ingresa usuario y contraseña.",
                elem_id="login-message",
                elem_classes=["error"],
            ),
            gr.update(),
        )

    ok, _ = verify(user, password)
    if not ok:
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
        gr.update(visible=False),
        gr.update(visible=True),
        gr.Markdown("", elem_id="login-message", elem_classes=["success"]),
        {"username": user, "ttl": time.time() + SESSION_TIMEOUT},
    )


def do_logout(user_id):
    """Handle user logout."""
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.Markdown("", elem_id="login-message"),
        {"username": None, "ttl": 0},
    )


def restore_session(stored_user: dict):
    """Restore user session on page load."""
    if (
        stored_user is None
        or stored_user.get("username") is None
        or stored_user.get("ttl", 0) < time.time()
    ):
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            {"username": None, "ttl": 0},
        )

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        f"Hola, **{stored_user.get('username')}** 👋",
        gr.update(),
    )


# Main Gradio Interface
with gr.Blocks(title="Chatbot GEMIS", theme=gemis_theme, css=custom_css) as demo:
    auth = gr.BrowserState({"username": None, "ttl": 0})

    # Login Screen
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
        with gr.Column(elem_id="login-card"):
            gr.Markdown("# Iniciar sesión")
            gr.HTML("<p class='subtitle'>Ingresá tus credenciales para continuar</p>")

            gr.Markdown("Nombre de usuario")
            user = gr.Textbox(
                elem_id="login-username",
                label="Nombre de usuario",
                value="",
                type="text",
                max_lines=1,
                container=False,
                elem_classes=["username-input"],
            )

            gr.Markdown("Contraseña")
            pwd = gr.Textbox(
                elem_id="login-password",
                label="Contraseña",
                type="password",
                lines=1,
                container=False,
            )

            login_btn = gr.Button(
                "Ingresar",
                elem_id="login-btn",
                variant="primary",
                size="md",
                icon="./assets/key-round.svg",
            )
            login_msg = gr.Markdown("", elem_id="login-message")

    # Main Chat Interface
    with gr.Blocks(fill_height=True):
        user_id = gr.State(None)
        thread_id = gr.State(None)
        message = gr.State(None)

        with gr.Column(visible=False) as sidebar_and_chat:
            with gr.Sidebar(width=420):
                with gr.Row(elem_id="sidebar-logo-container"):
                    gr.Image(
                        value="./assets/gemis-logo.png",
                        container=False,
                        show_download_button=False,
                        show_share_button=False,
                        show_fullscreen_button=False,
                        elem_id="sidebar-logo",
                    )
                new_btn = gr.Button(
                    "Nuevo",
                    size="sm",
                    variant="primary",
                    icon="./assets/square-pen.svg",
                )
                del_btn = gr.Button(
                    "Eliminar actual",
                    size="sm",
                    variant="stop",
                    icon="./assets/trash.svg",
                )
                gr.Markdown("### Tus Chats")
                chat_selector = gr.Radio(
                    choices=[],  # [(label, value), ...]
                    value=None,  # thread_id seleccionado
                    label=None,
                    interactive=True,
                    container=False,
                    elem_id="chat-list",  # para aplicar CSS estilo ChatGPT
                )

                gr.Markdown("### Archivos en este chat")
                files_table = gr.File(
                    label=None,
                    interactive=False,  # <- deshabilita subir/editar
                    file_count="multiple",  # muestra varios
                    show_label=False,
                    elem_id="files-view",
                    container=True,
                )

                gr.Markdown("---")
                logout_btn = gr.Button(
                    "Cerrar sesión",
                    size="md",
                    variant="secondary",
                    icon="./assets/log-out.svg",
                    elem_id="logout-btn",
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
                    placeholder="Escribí tu mensaje o subí archivos (PDF, max 10MB)...",
                    show_label=False,
                    file_types=[".pdf"],
                )

        # Event Handlers
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
        ).then(
            _reload_chats,
            inputs=[user_id],
            outputs=[chat_selector],
        )

        new_btn.click(
            _new_chat,
            inputs=[user_id],
            outputs=[chatbot, thread_id, chat_selector, files_table],
        )

        del_btn.click(
            _delete_chat,
            inputs=[user_id, thread_id],
            outputs=[chatbot, thread_id, chat_selector, files_table],
        )

        chat_selector.change(
            _switch_chat,
            inputs=[chat_selector, user_id],
            outputs=[chatbot, thread_id, files_table],
        )

        logout_btn.click(
            do_logout,
            inputs=[user_id],
            outputs=[login_column, sidebar_and_chat, login_msg, auth],
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
            lambda _: gr.Markdown("", elem_id="login-message"),
            inputs=[user],
            outputs=[login_msg],
        )

        pwd.change(
            lambda _: gr.Markdown("", elem_id="login-message"),
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
