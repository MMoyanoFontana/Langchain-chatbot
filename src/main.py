
import os
import yaml
from uuid import uuid4
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_community.document_loaders import PyMuPDFLoader
from src.splitter import split_docs
import gradio as gr
from src.vector_store import initialize_vector_store


from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, StateGraph, END



load_dotenv()


with open(os.path.join(os.path.dirname(__file__), "../config.yaml"), "r") as f:
    config = yaml.safe_load(f)

CHAT_MODEL = config.get("CHAT_MODEL")
EMBED_MODEL = config.get("EMBED_MODEL")
DATA_PATH = config.get("DATA_PATH", "test_data/ICA2609_ExamplesKE(2).pdf")
CHROMA_PATH = config.get("CHROMA_PATH", "chroma_db")
LANGSMITH_TRACING = config.get("LANGSMITH_TRACING", True)
LANGSMITH_PROJECT = config.get("LANGSMITH_PROJECT", "pr-formal-replacement-44")
LANGSMITH_ENDPOINT = config.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
TEMPERATURE = config.get("TEMPERATURE", 0.1)
print(f"Using CHAT_MODEL: {CHAT_MODEL}, EMBED_MODEL: {EMBED_MODEL}, LANGSMITH_TRACING: {LANGSMITH_TRACING}, TEMPERATURE: {TEMPERATURE}")
if not os.getenv("OPENAI_API_KEY") or (LANGSMITH_TRACING and not os.getenv("LANGSMITH_API_KEY")):
    raise ValueError("OPENAI_API_KEY and LANGSMITH_API_KEY must be set in .env file")

_embeddings = init_embeddings(EMBED_MODEL)
_vector_store = initialize_vector_store(_embeddings, storage_path=CHROMA_PATH)
_llm = init_chat_model(CHAT_MODEL, temperature=TEMPERATURE)
loader = PyMuPDFLoader(DATA_PATH)
docs = loader.load()
splits = split_docs(docs)

uuids = [str(uuid4()) for _ in range(len(splits))]
# adding chunks to vector store
_vector_store.add_documents(documents=splits, ids=uuids)


# Turn retrieva into a tool
@tool(response_format="content_and_artifact")
def retrieve(query: str, k: int = 8):
    """Retrieve information related to a query."""
    retrieved_docs = _vector_store.similarity_search(query, k=k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


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
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use five sentences maximum and keep the "
        "answer concise."
        "If possible add sources to your answer."
        "When ending, if a question arises, ask it to keep the user engaged."
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

    def _history_to_messages(history, user_msg):
        msgs = []
        for u, a in history:
            if u:
                msgs.append(HumanMessage(content=u))
            if a:
                msgs.append(AIMessage(content=a))
        msgs.append(HumanMessage(content=user_msg))
        return msgs

    def chat_fn(message_dict, history):
        """
        Called by gr.ChatInterface.
        - message_dict: dict with 'text' and 'file'
        - history: list of [user, assistant]
        """
        # 1) turn chat into MessagesState
        user_message = message_dict["text"]
        if message_dict["files"] and len(message_dict["files"]) > 0:
            return "Se subió un archivo. Pero aún no se procesa."
        msgs = _history_to_messages(history, user_message)
        result = _graph.invoke(
            {"messages": msgs}, config={"configurable": {"thread_id": "thread1"}}
        )
        ai_messages = result.get("messages", [])
        answer = ""
        if ai_messages:
            answer = getattr(ai_messages[-1], "content", "") or str(ai_messages[-1])
        return answer

    with gr.Blocks(title="Demo de RAG para Chatbot", theme=gr.themes.Base()) as demo:
        with gr.Sidebar():
            gr.Markdown("### Chats")
            gr.Button("Nuevo chat", variant="primary")
            gr.Markdown("---")
            gr.Button("Chat 1", variant="primary")
            gr.Button("Chat 2", variant="primary")
            gr.Button("Chat 3", variant="primary")

        chatbot = gr.ChatInterface(
            chat_fn,
            multimodal=True,
            chatbot=gr.Chatbot(
                height=512, label=None, show_label=False, container=False
            ),
            textbox=gr.MultimodalTextbox(
                placeholder="Haz una consulta o sube un documento...",
                file_types=[
                    ".pdf",
                ],
            ),
        )
    demo.launch()
