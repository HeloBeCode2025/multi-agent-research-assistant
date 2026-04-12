import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.graph import build_graph
#from config.settings import get_llm

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("May the kindly ones be with you")

question = st.text_area("Ask a question about autism research:")

if st.button("Research", type="primary"):
    if question:
        with st.spinner("Agents are working..."):
            app = build_graph()
            result = app.invoke({"question": question})
        
        st.header("Answer")
        st.markdown(result["draft"])
        
        with st.expander("Agent Trace"):
            st.subheader("Researcher's Notes")
            for note in result["research_notes"]:
                st.markdown(note.content)
            
            st.subheader("Critic's Review")
            st.markdown(result["critique"])
            st.metric("Revisions", result["revision_count"])