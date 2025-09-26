import os
import yaml
from dotenv import load_dotenv
import gradio as gr
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_community.document_loaders import PyMuPDFLoader
from parent_document_rag import build_parent_child_retriever
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, StateGraph, END
from langchain.schema import Document

load_dotenv()


with open(os.path.join(os.path.dirname(__file__), "../config.yaml"), "r") as f:
    config = yaml.safe_load(f)

CHAT_MODEL = config.get("CHAT_MODEL")
EMBED_MODEL = config.get("EMBED_MODEL")
DATA_PATH = config.get("DATA_PATH", "test_data/ICA2609_ExamplesKE(2).pdf")
LANGSMITH_TRACING = config.get("LANGSMITH_TRACING", True)
LANGSMITH_PROJECT = config.get("LANGSMITH_PROJECT", "pr-formal-replacement-44")
LANGSMITH_ENDPOINT = config.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
TEMPERATURE = config.get("TEMPERATURE", 0.1)
print(
    f"Using CHAT_MODEL: {CHAT_MODEL}, EMBED_MODEL: {EMBED_MODEL}, LANGSMITH_TRACING: {LANGSMITH_TRACING}, TEMPERATURE: {TEMPERATURE}"
)
if not os.getenv("OPENAI_API_KEY") or (
    LANGSMITH_TRACING and not os.getenv("LANGSMITH_API_KEY")
):
    raise ValueError("OPENAI_API_KEY and LANGSMITH_API_KEY must be set in .env file")

# Embeddings and LLM
_embeddings = init_embeddings(EMBED_MODEL)
_llm = init_chat_model(CHAT_MODEL, temperature=TEMPERATURE)

# Build parent-child retriever
_retriever = build_parent_child_retriever(
    pdf_path=DATA_PATH,
    embedding_fn=_embeddings,
)


# Turn parent-child retriever into a tool
@tool(response_format="content_and_artifact")
def retrieve(query: str, k: int = 4):
    """Retrieve information related to a query using parent-child retriever."""
    retrieved_docs = _retriever.invoke(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    # El segundo elemento debe ser una lista de objetos tipo dict
    docs_list = [
        doc.__dict__ if hasattr(doc, "__dict__") else doc for doc in retrieved_docs
    ]
    return serialized, docs_list


# Response that may include a tool call
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = _llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]
    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use ONLY the retrieved context to answer the question. "
        "If the context does not contain the answer, reply: "
        "'No encontré información relacionada a tu pregunta en los documentos.' "
        "or the equivalent in the user's language."
        "Keep the answer clear and concise. "
        "If possible add sources and pages to your answer in format [p.X-Y] "
        "When ending, if a question arises, ask it to keep the user engaged. "
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = _llm.invoke(prompt)
    return {"messages": [response]}


def compile_graph():
    graph_builder = StateGraph(MessagesState)
    tools = ToolNode([retrieve])
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()

    return graph_builder.compile(checkpointer=memory)


_graph = compile_graph()


if __name__ == "__main__":

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


custom_css = """
"""

with gr.Blocks(
    title="Demo de RAG para Chatbot",
    theme=gr.themes.Base(),
    css=custom_css,
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


if __name__ == "__main__":
    demo.launch()
