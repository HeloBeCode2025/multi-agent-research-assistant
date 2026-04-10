import sys
sys.path.append("..")

from config.settings import get_llm
from graph import ResearchState

def researcher_node(state: ResearchState) -> dict:
    llm = get_llm()
    prompt = f"Research the following question and provide key findings:\n{state['question']}"
    response = llm.invoke(prompt)
    return {"research_notes": [response], "revision_count": 0}