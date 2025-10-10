from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from nodes import retrieve, query_or_respond, generate
import sqlite3


def compile_graph():
    g = StateGraph(MessagesState)
    tools = ToolNode([retrieve])

    g.add_node(query_or_respond)
    g.add_node(tools)
    g.add_node(generate)

    g.set_entry_point("query_or_respond")
    g.add_conditional_edges(
        "query_or_respond", tools_condition, {END: END, "tools": "tools"}
    )
    g.add_edge("tools", "generate")
    g.add_edge("generate", END)

    conn = sqlite3.connect("chatbot_data.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return g.compile(checkpointer=checkpointer)
