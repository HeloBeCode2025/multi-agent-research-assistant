import sys
sys.path.append("..")

from config.settings import get_llm
from agents.state import ResearchState
from rag.retriever import get_retriever

def researcher_node(state: ResearchState) -> dict:
    retriever = get_retriever(k=3)
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = get_llm(temperature=0.3)
    prompt = f"""Research the following question using the provided context.
    
Context:
{context}

Question: {state['question']}

Provide key findings based on the context."""
    response = llm.invoke(prompt)
    return {"research_notes": [response], "revision_count": 0}