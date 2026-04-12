import sys
sys.path.append("..")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import get_llm

def create_summary_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant. Provide clear, concise summaries."),
        ("human", "Summarize the following topic in 5 bullet points:\n\n{topic}")
    ])
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    return chain