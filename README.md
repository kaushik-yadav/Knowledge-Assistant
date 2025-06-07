# 📚 Knowledge Assistant (RAG based)

An interactive, multi-agent Q\&A assistant built using LangChain, FAISS, and a large language model (LLM). It can define terms, solve calculations, and retrieve answers from documents (including user-uploaded ones) , all via a clean Streamlit interface.

---

## 🔧 Features

* **Streamlit Frontend**
  Upload `.txt`/`.pdf` files, ask questions, and get tool-specific responses with context previews, all from a single UI.

* **Multi-Agent Reasoning**
  Auto-selects the right tool based on the query:

  * **CALCULATE**: Performs math.
  * **DEFINE**: Gets definitions via Oxford API.
  * **RAG**: Retrieves document-based answers (see below).

* **Document Uploads + RAG**
  Users can upload `.txt`/`.pdf` files directly via the sidebar. These files are chunked, embedded (using `all-MiniLM-L6-v2`), and indexed into FAISS in real-time. Queries then search across **both preloaded and uploaded** content.

* **LangChain Agents + Memory**
  Uses `initialize_agent` with `ChatOpenAI`, `LLMChecker`, and **ConversationBufferMemory** to track ongoing Q\&A history. This gives answers more context across turns.

---

## 🚀 Quickstart

```bash
git clone https://github.com/kaushik-yadav/knowledge-assistant.git
cd knowledge-assistant

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

---

## 🔑 Secrets & Config

You can store your keys in **`.streamlit/secrets.toml`** (used on both Streamlit Cloud and locally):

```toml
OPENAI_API_KEY = "your_openai_or_togetherai_key"
OPENAI_API_BASE = "https://api.together.xyz/v1"

[OXFORD]
APP_ID  = "your_oxford_app_id"
APP_KEY = "your_oxford_app_key"
```

Alternatively, you can set environment variables:

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_API_BASE="your_base"
export OXFORD_APP_ID="your_app_id"
export OXFORD_APP_KEY="your_app_key"
```

---

## 📁 Project Structure

```
knowledge-assistant/
├──.streamlit               # Streamlit credentials/secrets
├── docs/                   # Preloaded .txt files
├── src/
│   ├── llm.py              # Tool routing logic
│   └── rag.py              # FAISS + retriever setup
├── .gitignore              
├── app.py                  # Streamlit UI logic
├── constants.py            # Constants & paths
├── main.py                 # CLI mode (optional)
├── requirements.txt
├── utils.py                # Chunking & loader functions
└── data/                   # FAISS index files (after running the code)
```

---

## 💡 Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature/xyz`)
3. Commit (`git commit -m "feat: xyz"`)
4. Push + PR
