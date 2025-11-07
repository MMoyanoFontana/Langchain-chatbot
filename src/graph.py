# Multi-user Chroma (Option A): single collection + per-user/per-thread filters
# - Persistent vector store (no ingestion here)
# - Tool enforces {user_id, thread_id} from server context (not user input)
# - Sistema de memorias de usuario
# - Plug-and-play with your existing LangGraph workflow

import sqlite3
import dotenv
import json
from typing import List, Literal, Dict, Any

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_chroma import Chroma
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage import LocalFileStore
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import create_kv_docstore
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

dotenv.load_dotenv()

PERSIST_DIR = "./chroma_multi"
_embeddings = OpenAIEmbeddings()
RESPONSE_MODEL = init_chat_model("openai:gpt-4o", temperature=0)
GRADER_MODEL = init_chat_model("openai:gpt-4o", temperature=0)
MEMORY_MODEL = init_chat_model("openai:gpt-4o", temperature=0)
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

# -----------------------------
# Sistema de Memorias de Usuario
# ----------------------------

# Inicializar base de datos de memorias
def init_memory_db():
    """Inicializa la base de datos SQLite para memorias de usuario."""
    conn = sqlite3.connect("user_memories.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_memories (
            user_id INTEGER,
            memory_key TEXT,
            memory_value TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, memory_key)
        )
    """)
    conn.commit()
    return conn


memory_conn = init_memory_db()


class Memory(BaseModel):
    """Una memoria individual del usuario."""

    key: str = Field(
        description="Clave descriptiva de la memoria (ej: 'nombre', 'equipo_favorito', 'profesion')"
    )
    value: str = Field(description="Valor de la memoria")


class UserMemory(BaseModel):
    """Estructura para extraer memorias del usuario."""

    has_memories: bool = Field(
        description="True si se encontraron memorias personales en el mensaje, False si no."
    )
    memories: List[Memory] = Field(
        default_factory=list,
        description="Lista de memorias extraídas. Vacía si has_memories es False.",
    )


MEMORY_EXTRACTION_PROMPT = """Analiza el siguiente mensaje del usuario y extrae información personal relevante que deba recordarse.

Mensaje del usuario: {message}

Extrae información como:
- Nombre
- Preferencias (equipos deportivos, comidas, hobbies)
- Profesión u ocupación
- Información familiar
- Ubicación
- Cualquier dato personal relevante que el usuario comparta

Si el mensaje NO contiene información personal que deba recordarse, marca has_memories como False y deja memories vacía.
Si SÍ contiene información personal, extrae las memorias con claves descriptivas en español.

Ejemplos:
- "Me llamo Juan" → key: "nombre", value: "Juan"
- "Me gusta Boca" → key: "equipo_favorito", value: "Boca"
- "Soy ingeniero y vivo en Buenos Aires" → [key: "profesion", value: "ingeniero"], [key: "ciudad", value: "Buenos Aires"]
"""


def extract_and_save_memories(message: str, user_id: int):
    """Extrae memorias del mensaje del usuario y las guarda en la base de datos."""
    prompt = MEMORY_EXTRACTION_PROMPT.format(message=message)

    response = MEMORY_MODEL.with_structured_output(UserMemory).invoke(
        [{"role": "user", "content": prompt}]
    )

    if response.has_memories and response.memories:
        cursor = memory_conn.cursor()
        saved_memories = {}
        for memory in response.memories:
            cursor.execute(
                """
                INSERT OR REPLACE INTO user_memories (user_id, memory_key, memory_value)
                VALUES (?, ?, ?)
            """,
                (user_id, memory.key, memory.value),
            )
            saved_memories[memory.key] = memory.value
        memory_conn.commit()
        return saved_memories

    return None


def get_user_memories(user_id: int) -> Dict[str, str]:
    """Recupera todas las memorias de un usuario."""
    cursor = memory_conn.cursor()
    cursor.execute(
        """
        SELECT memory_key, memory_value 
        FROM user_memories 
        WHERE user_id = ?
    """,
        (user_id,),
    )

    memories = {}
    for row in cursor.fetchall():
        memories[row[0]] = row[1]

    return memories


def format_memories_for_context(memories: Dict[str, str]) -> str:
    """Formatea las memorias para incluirlas en el contexto del modelo."""
    if not memories:
        return ""

    memory_text = "Información que conozco sobre ti:\n"
    for key, value in memories.items():
        memory_text += f"- {key.replace('_', ' ').capitalize()}: {value}\n"

    return memory_text


# -----------------------------
# Tools
# -----------------------------


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


# -----------------------------
# Nodos del Grafo
# -----------------------------


def extract_memories_node(state: MessagesState, config: RunnableConfig):
    """Extrae y guarda memorias del último mensaje del usuario."""
    user_id = int(config["configurable"].get("user_id"))

    # Obtener el último mensaje del usuario
    last_user_message = None
    for m in reversed(state["messages"]):
        if m.type == "human":
            last_user_message = m.content
            break

    if last_user_message:
        extracted = extract_and_save_memories(last_user_message, user_id)
        if extracted:
            print(f"Memorias extraídas para usuario {user_id}: {extracted}")

    return state


def generate_query_or_respond(state: MessagesState, config: RunnableConfig):
    """Ask the model; it will decide to call the retriever tool or answer directly."""
    user_id = int(config["configurable"].get("user_id"))

    # Obtener memorias del usuario
    memories = get_user_memories(user_id)
    memory_context = format_memories_for_context(memories)

    # Agregar contexto de memorias al mensaje del sistema si existen
    messages = state["messages"].copy()
    if memory_context:
        system_message = {
            "role": "system",
            "content": f"{memory_context}\nUsa esta información para personalizar tus respuestas cuando sea relevante.",
        }
        messages.insert(0, system_message)

    response = RESPONSE_MODEL.bind_tools([retriever_tool]).invoke(messages)
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


def generate_answer(state: MessagesState, config: RunnableConfig):
    """Generate an answer with user memories context."""
    user_id = int(config["configurable"].get("user_id"))

    question = ""
    for m in reversed(state["messages"]):
        if m.type == "human":
            question = m.content
            break

    context = _last_tool_payload(state["messages"])

    # Agregar memorias del usuario al contexto
    memories = get_user_memories(user_id)
    memory_context = format_memories_for_context(memories)

    full_context = context
    if memory_context:
        full_context = f"{memory_context}\n\nDocumentos recuperados:\n{context}"

    prompt = GENERATE_PROMPT.format(question=question, context=full_context)
    response = RESPONSE_MODEL.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# -----------------------------
# Graph
# -----------------------------
workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node("extract_memories", extract_memories_node)
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

# Primero extraemos memorias
workflow.add_edge(START, "extract_memories")
workflow.add_edge("extract_memories", "generate_query_or_respond")

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


# -----------------------------
# Funciones auxiliares para gestionar memorias
# -----------------------------


def delete_user_memory(user_id: int, memory_key: str):
    """Elimina una memoria específica de un usuario."""
    cursor = memory_conn.cursor()
    cursor.execute(
        """
        DELETE FROM user_memories 
        WHERE user_id = ? AND memory_key = ?
    """,
        (user_id, memory_key),
    )
    memory_conn.commit()


def clear_all_user_memories(user_id: int):
    """Elimina todas las memorias de un usuario."""
    cursor = memory_conn.cursor()
    cursor.execute(
        """
        DELETE FROM user_memories 
        WHERE user_id = ?
    """,
        (user_id,),
    )
    memory_conn.commit()
