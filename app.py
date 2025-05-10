import streamlit as st

from src.llm import run_llm_agent
from src.rag import start_rag


def main():
    st.title("Knowledge Assistant")
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
