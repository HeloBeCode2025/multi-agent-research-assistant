import sys
sys.path.append("..")

from config.settings import get_llm
from graph import ResearchState

def critic_node(state: ResearchState) -> dict:
    llm = get_llm()
    prompt = f"""Review this draft and decide if it's good enough.
    
Draft:
{state['draft']}

Respond with EXACTLY one of:
- "APPROVED" if the draft is comprehensive and well-written
- "REVISE: <specific feedback>" if it needs improvement
"""
    response = llm.invoke(prompt)
    content = response.content.strip()
    is_approved = content.startswith("APPROVED")
    return {
        "critique": content,
        "is_approved": is_approved,
        "revision_count": state["revision_count"] + 1
    }