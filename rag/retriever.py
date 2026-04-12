from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import chromadb

def get_retriever(k: int = 3):
    embeddings = OllamaEmbeddings(model="llama3.2:3b")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        collection_name="research_articles",
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})