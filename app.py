import os

import streamlit as st

from config import DOCS_PATH
from src.llm import run_llm_agent
from src.rag import build_retriever_from_uploaded_text, extract_text_from_pdf, start_rag

# Load Secrets from Streamlit's Secret Manager
try:
    OPENAI_API_KEY  = st.secrets["OPENAI_API_KEY"]
    OPENAI_API_BASE = st.secrets["OPENAI_API_BASE"]
    OXFORD_APP_ID   = st.secrets["OXFORD"]["APP_ID"]
    OXFORD_APP_KEY  = st.secrets["OXFORD"]["APP_KEY"]
except KeyError as e:
    st.error(f"Missing secret: {e}")
    st.stop()


# set these as environment variables
def load_secrets():
    try:
        os.environ["OPENAI_API_KEY"]  = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]
        os.environ["OXFORD_APP_ID"]   = st.secrets["OXFORD"]["APP_ID"]
        os.environ["OXFORD_APP_KEY"]  = st.secrets["OXFORD"]["APP_KEY"]
    except KeyError as e:
        st.error(f"Missing secret: {e}")
        st.stop()



def main():
    load_secrets()
    st.title("📚 Knowledge Assistant")

    # SIDEBAR: Inspect Raw Documents
    st.sidebar.header("📁 Inspect Raw Sample Documents")
    txt_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".txt")]
    selected = st.sidebar.selectbox("Choose a file", [""] + txt_files)
    if selected:
        path = os.path.join(DOCS_PATH, selected)
        with open(path, "r", encoding="utf-8") as f:
            st.sidebar.text(f.read())

    # MAIN: File Upload
    st.subheader("📤 Upload Your Own File (.txt or .pdf)")
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"])

    # Question Input
    query = st.text_input("💬 Ask a question:")
    if not query:
        return

    # Choose Retriever Source
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            user_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            user_text = extract_text_from_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return
        retriever = build_retriever_from_uploaded_text(user_text)
    else:
        if "retriever" not in st.session_state:
            st.session_state.retriever = start_rag()
        retriever = st.session_state.retriever

    # Run LLM Agent
    result = run_llm_agent(retriever, query)

    st.subheader("🧠 Final Answer")
    st.write(result["answer"])

    st.subheader("🛠️ Tool/Agent Branch Used")
    st.write(result["tool_used"])

    if result["context_snippets"]:
        st.subheader("📌 Retrieved Context Snippets")
        for snippet in result["context_snippets"]:
            st.write(snippet)


if __name__ == "__main__":
    main()
