import os

import streamlit as st

from config import DOCS_PATH

# loading secrets from Streamlit’s Secrets Manager
OPENAI_API_KEY  = st.secrets["OPENAI_API_KEY"]
OPENAI_API_BASE = st.secrets["OPENAI_API_BASE"]
OXFORD_APP_ID  = st.secrets["OXFORD"]["APP_ID"]
OXFORD_APP_KEY = st.secrets["OXFORD"]["APP_KEY"]


# optionally setting it through environment
os.environ["OPENAI_API_KEY"]  = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
os.environ["OXFORD_APP_ID"]   = OXFORD_APP_ID
os.environ["OXFORD_APP_KEY"]  = OXFORD_APP_KEY


from src.llm import run_llm_agent
from src.rag import start_rag


def main():
    st.title("Knowledge Assistant")
    # sidebar for inspecting the raw documents
    st.sidebar.header("Inspect Raw Documents")
    txt_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".txt")]
    selected = st.sidebar.selectbox("Choose a file", [""] + txt_files)
    if selected:
        path = os.path.join(DOCS_PATH, selected)
        with open(path, "r", encoding="utf-8") as f:
            st.sidebar.text(f.read())

    query = st.text_input("Ask a question:")
    if not query:
        return

    # Initialize retriever once per session
    if "retriever" not in st.session_state:
        st.session_state.retriever = start_rag()

    result = run_llm_agent(st.session_state.retriever, query)

    st.subheader("Tool/Agent Branch Used")
    st.write(result["tool_used"])

    if result["context_snippets"]:
        st.subheader("Retrieved Context Snippets")
        for snippet in result["context_snippets"]:
            st.write(snippet)

    st.subheader("Final Answer")
    st.write(result["answer"])

if __name__ == "__main__":
    main()
