import sys
sys.path.append("..")

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config.settings import get_llm
from agents.state import ResearchState

class CritiqueResult(BaseModel):
    """Structured output from the Critic agent."""
    is_approved: bool = Field(
        description="Whether the draft is good enough to finalize"
    )
    feedback: str = Field(
        description="Specific, actionable feedback for the Writer. If approved, briefly explain why."
    )
    completeness: int = Field(
        description="Score 1-10 for how thoroughly the draft answers the question",
        ge=1, le=10,
    )
    accuracy: int = Field(
        description="Score 1-10 for factual accuracy based on the research notes",
        ge=1, le=10,
    )
    clarity: int = Field(
        description="Score 1-10 for writing quality and clarity",
        ge=1, le=10,
    )

def critic_node(state: ResearchState) -> dict:
    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(CritiqueResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a rigorous research critic. Review the draft below \
and evaluate it on completeness, accuracy, and clarity. Approve only if all \
three scores are at least 7. Provide specific, actionable feedback."""),
        ("human", """Original question: {question}

Draft to review:
{draft}

Research notes the draft should be based on:
{notes}""")
    ])

    chain = prompt | structured_llm

    notes = "\n".join([msg.content for msg in state["research_notes"]])
    result: CritiqueResult = chain.invoke({
        "question": state["question"],
        "draft": state["draft"],
        "notes": notes,
    })
    
    return {
    "critique": result.feedback,
    "is_approved": result.is_approved,
    "revision_count": state["revision_count"] + 1,
    "completeness_score": result.completeness,
    "accuracy_score": result.accuracy,
    "clarity_score": result.clarity,
}