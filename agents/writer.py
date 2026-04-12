import sys
sys.path.append("..")

from config.settings import get_llm
from agents.state import ResearchState

def writer_node(state: ResearchState) ->dict:
    llm = get_llm()
    notes = "\n".join([msg.content for msg in state["research_notes"]])
    prompt = f"Write a comprehensive answer based on these research notes:\n{notes}"
    response = llm.invoke(prompt)
    print(f"[WRITER] Draft length: {len(response.content)} chars")
    return {"draft": response.content}