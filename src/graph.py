# Multi-user Chroma (Option A): single collection + per-user/per-thread filters
# - Persistent vector store (no ingestion here)
# - Tool enforces {user_id, thread_id} from server context (not user input)
# - Plug-and-play with your existing LangGraph workflow

import sqlite3
import dotenv
from typing import List, Literal

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_chroma import Chroma
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage._lc_store import create_kv_docstore
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

dotenv.load_dotenv()

PERSIST_DIR = "./chroma_multi"
_embeddings = OpenAIEmbeddings()
RESPONSE_MODEL = init_chat_model("openai:gpt-4o", temperature=0)
GRADER_MODEL = init_chat_model("openai:gpt-4o", temperature=0)
COLLECTION_NAME = "child_store"
vs = Chroma(
    embedding_function=_embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
)

fs = LocalFileStore("./parent")
store = create_kv_docstore(fs)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=800)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=60)

retriever = ParentDocumentRetriever(
    vectorstore=vs,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)


@tool
def retriever_tool(query: str, config: RunnableConfig, k: int = 4) -> List[Document]:
    """
    Retrieve relevant documents for the current request's user/thread.
    NOTE: user_id/thread_id are resolved from server context, not from user input.
    """
    filter = {
        "$and": [
            {"user_id": {"$eq": int(config["configurable"].get("user_id"))}},
            {"thread_id": {"$eq": str(config["configurable"].get("thread_id"))}},
        ]
    }
    retriever.search_kwargs["filter"] = filter
    retriever.search_kwargs["k"] = k
    return retriever.invoke(query)


def generate_query_or_respond(state: MessagesState):
    """Ask the model; it will decide to call the retriever tool or answer directly."""
    # Bind the tool that already enforces server-side user/thread filters
    response = RESPONSE_MODEL.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def _last_tool_payload(msgs: List[BaseMessage]) -> str:
    for m in reversed(msgs):
        if isinstance(m, ToolMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    # latest user message
    question = ""
    for m in reversed(state["messages"]):
        if m.type == "human":
            question = m.content
            break
    context = _last_tool_payload(state["messages"])

    prompt = GRADE_PROMPT.format(question=question, context=context)
    resp = GRADER_MODEL.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    return (
        "generate_answer"
        if resp.binary_score.lower().strip() == "yes"
        else "rewrite_question"
    )


REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = RESPONSE_MODEL.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}


GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "Context: {context}"
    "Question: {question} \n"
    "If you don't know the answer, just say that you don't know. "
    "Answer in Spanish unless explicitly asked to answer in another language. "
    "Use five sentences maximum and keep the answer concise.\n"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = ""
    for m in reversed(state["messages"]):
        if m.type == "human":
            question = m.content
            break
    context = _last_tool_payload(state["messages"])
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = RESPONSE_MODEL.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# -----------------------------
# Graph
# -----------------------------
workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
conn = sqlite3.connect("chat.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = workflow.compile(checkpointer=checkpointer)
