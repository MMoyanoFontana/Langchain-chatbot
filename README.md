# Langchain Chatbot

Chatbot en progreso para **Retrieval Augmented Generation (RAG) y preguntas y respuestas sobre documentos** usando **LangChain y LangGraph ** y un frontend con **Gradio** para probar rápidamente.

> **Objetivo:** permitir que un usuario cargue un documento, se indexe en un vector store y hacer preguntas al LLM con recuperación de pasajes relevantes.

---

## Demo rápida

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

## Funcionamiento

1. Carga de documento → El usuario sube un archivo.

2. Particionado (chunking) → El documento se divide en fragmentos

3. Generación de embeddings → Cada fragmento se transforma en un vector numérico.

4. Almacenamiento en Vector Store → Los vectores se indexan para búsquedas eficientes.

5. Consulta del usuario → El usuario realiza una pregunta.

6. Recuperación de pasajes relevantes → Se buscan los fragmentos más útiles en el Vector Store.

7. Respuesta con LLM → El modelo LLM responde usando los fragmentos recuperados como contexto.

## Modelos actuales

Actualmente se utilizan los siguientes modelos:

- **Modelo de embeddings:**
  - `embeddinggemma:300m`
  - Se ejecuta de forma local utilizando Ollama.
  - Convierte los fragmentos de texto en vectores numéricos para la búsqueda semántica en el vector store.

- **Modelo de chat (LLM):**
  - `llama-3.1-8b-instant`
  - Se utiliza a través de [Groq](https://groq.com/).
  - Se encarga de generar respuestas a las preguntas del usuario usando los fragmentos relevantes recuperados del documento.


Estos modelos se pueden cambiar editando las variables `CHAT_MODEL` y `EMBED_MODEL` en tu entorno.