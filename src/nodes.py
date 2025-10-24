from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage
from setup import _retriever, _llm
from langchain_core.tools import tool


@tool(response_format="content_and_artifact")
def retrieve(query: str, config: RunnableConfig, k: int = 4):
    """Retrieve information related to a query using parent-child retriever."""
    _retriever.search_kwargs = {
        "k": k,
        "filter": {
            "$and": [{"user_id": {"$eq": int(config["configurable"].get("user_id"))}}, 
                     {"thread_id": {"$eq": str(config["configurable"].get("thread_id"))}}]
        },
    }
    retrieved_docs = _retriever.invoke(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    docs_list = [
        doc.__dict__ if hasattr(doc, "__dict__") else doc for doc in retrieved_docs
    ]
    return serialized, docs_list


def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = _llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


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
        """You are an assistant specialized in question-answering tasks.  
        Rely **exclusively** on the retrieved context to respond.  
        If the context does not contain the answer, reply:  
        “No encontré información relacionada con tu pregunta en los documentos.”  
        Keep answers **clear, direct, and concise**.  
        When possible, include sources and page ranges in the format [p.X-Y].  
        Use LaTeX delimiters `$$ ... $$` for mathematical expressions.
        """
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = _llm.invoke(prompt)
    return {"messages": [response]}
