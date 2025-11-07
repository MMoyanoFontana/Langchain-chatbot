from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
_title_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


def _generate_title_openai(user_text: str, max_len: int = 60, lang: str = "es") -> str:
    """
    Pide a OpenAI un título corto estilo ChatGPT (4-6 palabras, sin comillas).
    Fallback: usa un recorte del primer renglón si algo falla.
    """
    prompt_sys = (
        "Eres un asistente que propone títulos concisos y útiles para conversaciones. "
        "Si el texto comienza con 'files: ', genera un título basado en los nombres de los archivos. "
        "Responde únicamente con el título, sin comillas ni puntuación al final. "
        "Usa 4-6 palabras, en mayúsculas/minúsculas naturales, y en el mismo idioma del usuario."
    )
    try:
        resp = _title_llm.invoke(
            [
                SystemMessage(content=prompt_sys),
                HumanMessage(
                    content=(
                        f"Genera un título breve (máx {max_len} caracteres) para este primer mensaje. "
                        f"Idioma: {lang}. Texto:\n\n{user_text}"
                    )
                ),
            ]
        )
        title = (resp.content or "").strip()
        # limpiar comillas accidentales y truncar
        title = title.strip("“”\"'").strip()
        if len(title) > max_len:
            title = title[:max_len].rstrip() + "…"
        if not title:
            raise ValueError("Título vacío")
        return title
    except Exception as _:
        # Fallback ultra simple
        first = user_text.strip().splitlines()[0][:max_len].strip()
        if not first:
            first = "Nuevo chat"
        return first
