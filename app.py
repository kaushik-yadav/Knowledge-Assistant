import os

import streamlit as st

from constants import DOCS_PATH
from src.llm import run_llm_agent
from src.rag import build_retriever_from_uploaded_text, extract_text_from_pdf, start_rag

# === Load Secrets or Environment Variables ===
# This checks if you're running on Streamlit Cloud (st.secrets) or locally (os.environ)
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    OPENAI_API_BASE = st.secrets["OPENAI_API_BASE"]
    OXFORD_APP_ID = st.secrets["OXFORD"]["APP_ID"]
    OXFORD_APP_KEY = st.secrets["OXFORD"]["APP_KEY"]

except Exception:
    # Fall back to environment variables if st.secrets is not available
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OXFORD_APP_ID = os.getenv("OXFORD_APP_ID")
    OXFORD_APP_KEY = os.getenv("OXFORD_APP_KEY")

# === Stop the app if any key is missing ===
if not all([OPENAI_API_KEY, OPENAI_API_BASE, OXFORD_APP_ID, OXFORD_APP_KEY]):
    st.error("Missing API keys. Please check your environment variables or secrets.")
    st.stop()

# Set the keys as environment variables (so other code can access them)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE
os.environ["OXFORD_APP_ID"] = OXFORD_APP_ID
os.environ["OXFORD_APP_KEY"] = OXFORD_APP_KEY

# === MAIN APP ===
def main():
    st.title("📚 Knowledge Assistant")

    # Let user choose between RAG or LLM-only answers
    st.sidebar.header("Settings")
    use_rag = st.sidebar.checkbox(
        "Use sample documents for context (RAG)?", 
        value=True
    )

    # Let user view sample documents (RAG files)
    st.sidebar.subheader("📁 Inspect Sample Documents")
    txt_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".txt")]
    selected = st.sidebar.selectbox("Choose a file", [""] + txt_files)
    if selected:
        with open(os.path.join(DOCS_PATH, selected), "r", encoding="utf-8") as f:
            st.sidebar.text(f.read())

    # File upload section
    st.subheader("📤 Upload Your Own File (.txt or .pdf)")
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"])

    # User question input
    query = st.text_input("💬 Ask a question:")
    if not query:
        return

    # Build retriever if RAG is enabled
    retriever = None
    if use_rag:
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

    # Run the AI assistant
    result = run_llm_agent(
        retriever=retriever,
        query=query,
        session_id="default",
        use_rag=use_rag
    )

    # Show the results
    st.subheader("🧠 Final Answer")
    st.write(result["answer"])

    st.subheader("🛠️ Tool/Agent Branch Used")
    st.write(result["tool_used"])

    if result.get("context_snippets"):
        st.subheader("📌 Retrieved Context Snippets")
        for snippet in result["context_snippets"]:
            st.write(snippet)


if __name__ == "__main__":
    main()
