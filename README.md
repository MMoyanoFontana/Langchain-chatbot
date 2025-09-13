# Langchain Chatbot

Chatbot en progreso para **RAG y QA sobre documentos** usando **LangChain y LangGraph ** y un front **Gradio** para probar rápidamente en el navegador.

> **Objetivo:** permitir que un usuario cargue un documento, se indexe en un vector store y hacer preguntas al LLM con recuperación de pasajes relevantes.

---

## 🚀 Demo rápida

1. Clonar
```bash
git clone https://github.com/MMoyanoFontana/Langchain-chatbot
cd Langchain-chatbot
```

2. (Opción A) Instalar con uv (recomendado)
```bash
#   uv: https://github.com/astral-sh/uv
uv sync
```

2. (Opción B) Instalar con pip
```bash
# crear venv y actívarlo, luego:
pip install -r requirements.txt
```
3. Definir variables de entorno
```bash
#   ejemplo en Linux/macOS:
export HOST_IP=172.17.32.1
export OLLAMA_BASE_URL="http://$HOST_IP:11434"
export CHAT_MODEL="llama-3.1-8b-instant"
export EMBED_MODEL="embeddinggemma:300m"
```
4. Ejecuta
```bash
uv run python main.py
# o
python main.py
```
