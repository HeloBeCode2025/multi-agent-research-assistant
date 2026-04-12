from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages


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