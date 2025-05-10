import os

import streamlit as st

from config import DOCS_PATH
from src.llm import run_llm_agent
from src.rag import start_rag


def main():
    st.title("Knowledge Assistant")

    st.sidebar.header("Inspect Raw Documents")
    txt_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".txt")]
    selected_file = st.sidebar.selectbox("Choose a file", [""] + txt_files)
    if selected_file:
        file_path = os.path.join(DOCS_PATH, selected_file)
        with open(file_path, "r", encoding="utf-8") as f:
            st.sidebar.subheader(selected_file)
            st.sidebar.text(f.read())

    query = st.text_input("Ask a question:")

    if query:
        retriever = start_rag()
        result = run_llm_agent(retriever, query)

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
