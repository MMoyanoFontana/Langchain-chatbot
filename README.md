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
3. Definir en el config.yaml los modelos a utilizar. Ejemplo:
```yaml
EMBED_MODEL: "openai:text-embedding-3-large"
CHAT_MODEL: "openai:gpt-4o-mini"
DATA_PATH: "test_data/ICA2609_ExamplesKE(2).pdf"
CHROMA_PATH: "chroma_db"
LANGSMITH_TRACING: True
LANGSMITH_PROJECT: "pr-formal-replacement-44"
LANGSMITH_ENDPOINT: "https://api.smith.langchain.com"
TEMPERATURE: 0.2
``` 
4. Definir como variables variables de entorno las key necesarias
```bash
#   ejemplo:
OPENAI_API_KEY=
LANGSMITH_API_KEY=
```
4. Ejecutar
```bash
uv run python -m src.main 
o
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
  - `openai:text-embedding-3-large`
  - Convierte los fragmentos de texto en vectores numéricos para la búsqueda semántica en el vector store.

- **Modelo de chat (LLM):**
  - `openai:gpt-4o-mini`
  - Se encarga de generar respuestas a las preguntas del usuario usando los fragmentos relevantes recuperados del documento.


Estos se pueden cambiar  modificando el archivo config.yaml con el formato `proveedor:modelo`
