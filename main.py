import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import gradio as gr

load_dotenv()

HOST_IP = os.getenv("HOST_IP")
OLLAMA_BASE_URL = f"http://{HOST_IP}:11434"
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")

if not CHAT_MODEL or not EMBED_MODEL:
    raise ValueError("CHAT_MODEL and EMBED_MODEL must be set in .env file")

_embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

vector_size = len(_embeddings.embed_query("sample text"))
client = QdrantClient(":memory:")
if not client.collection_exists("test"):
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
_vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=_embeddings,
)


def load_and_split_text():
    loader = PyPDFLoader("test_data/ICA2609_ExamplesKE.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    _ = _vector_store.add_documents(documents=splits)
    return


_llm = init_chat_model(model=CHAT_MODEL, model_provider="groq")
# _llm = init_chat_model(
#    "llama3.1:8b-instruct-q4_K_M",
#    model_provider="ollama",
#    base_url=OLLAMA_BASE_URL,
# )


# Turn retrieva into a tool
@tool(response_format="content_and_artifact")
def retrieve(query: str, k: int = 4):
    """Retrieve information related to a query."""
    retrieved_docs = _vector_store.similarity_search(query, k=k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Generate a response that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = _llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
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

    # Run
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


load_and_split_text()
_graph = compile_graph()


def _history_to_messages(history, user_msg):
    msgs = []
    for u, a in history:
        if u:
            msgs.append(HumanMessage(content=u))
        if a:
            msgs.append(AIMessage(content=a))
    msgs.append(HumanMessage(content=user_msg))
    return msgs


def chat_fn(user_message, history):
    """
    Called by gr.ChatInterface.
    - user_message: latest user text
    - history: list of [user, assistant]
    """
    # 1) turn chat into MessagesState
    msgs = _history_to_messages(history, user_message)
    result = _graph.invoke(
        {"messages": msgs}, config={"configurable": {"thread_id": "thread1"}}
    )
    ai_messages = result.get("messages", [])
    answer = ""
    if ai_messages:
        answer = getattr(ai_messages[-1], "content", "") or str(ai_messages[-1])
    return answer


with gr.Blocks(title="Chat Demo") as demo:
    chatbot = gr.Chatbot(height=420, show_copy_button=True)
    chat = gr.ChatInterface(
        fn=chat_fn,
        chatbot=chatbot,
        textbox=gr.Textbox(placeholder="Ask something about the PDF…"),
    )

if __name__ == "__main__":
    demo.launch()
