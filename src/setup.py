from parent_document_rag import build_parent_child_retriever
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
import yaml
import os

load_dotenv()

# Cargar configuración desde config.yaml
with open(os.path.join(os.path.dirname(__file__), "../config.yaml"), "r") as f:
    config = yaml.safe_load(f)

DATA_PATH = config.get("DATA_PATH")
CHAT_MODEL = config.get("CHAT_MODEL")
EMBED_MODEL = config.get("EMBED_MODEL")
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

# Inicializar embeddings y modelo de chat
_embeddings = init_embeddings(EMBED_MODEL)
_llm = init_chat_model(CHAT_MODEL, temperature=TEMPERATURE)

# Inicializar el retriever
_retriever = build_parent_child_retriever(
    pdf_path=None,
    embedding_fn=_embeddings,
)
