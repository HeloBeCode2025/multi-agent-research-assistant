import sys
sys.path.append("..")

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from agents.researcher import researcher_node
from agents.writer import writer_node
from agents.critic import critic_node

class ResearchState(TypedDict):
    question: str
    research_notes: Annotated[list, add_messages]
    draft: str
    critique: str
    revision_count: int
    is_approved: bool
    completeness_score: int
    accuracy_score: int
    clarity_score: int

def should_continue(state: ResearchState) -> str:
    """Route based on critic's verdict and revision count."""
    if state["is_approved"]:
        return "end"
    if state["revision_count"] >= 3: # Safety limit
        return "end"
    return "revise"

def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic",
        should_continue,
        {"revise": "writer", "end": END}
    )

    return graph.compile()

#result = app.invoke({"question": "Why should researcher listen to autistic people?"})
#print(result["draft"])