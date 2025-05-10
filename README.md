# Knowledge Assistant: RAG-Powered Multi-Agent Q\&A

A Retrieval-Augmented Generation (RAG) assistant built with LangChain, FAISS, and a 70B-parameter LLM to answer questions by dynamically choosing between calculation, definition (via Oxford Dictionary), or context retrieval. Includes a Streamlit interface for browsing raw documents, viewing which tool was used, inspecting retrieved snippets, and reading concise answers.

---

## Features

* **Document Ingestion & Chunking**
  Raw `.txt` files in `docs/` are split into semantic chunks (FAQs, product specs, company overviews).

* **Vector Index & Retrieval**
  Embeds chunks with Hugging Face’s `all-MiniLM-L6-v2` model and indexes them in FAISS for fast top-k semantic search.

* **Agentic Workflow**

  1. **CALCULATE**: LLM self-identifies arithmetic queries and performs the computation.
  2. **DEFINE**: LLM extracts the term, fetches its definition via the Oxford API, then formats the answer per request.
  3. **RAG**: Retrieves relevant document snippets via FAISS and answers using context through LangChain’s retrieval chain.

* **LLM Integration**
  Uses `ChatOpenAI` from `langchain_openai` to communicate with a 70B-parameter model, wrapped in modular code.

* **Streamlit Demo UI**

  * **Sidebar**: Browse and view raw source documents.
  * **Main Panel**: Enter questions and see which tool ran, view retrieved context, and read the final answer.

---

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/kaushik-yadav/knowledge-assistant.git
   cd knowledge-assistant
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # macOS/Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Configuration

### Secrets Management

On Streamlit Cloud, go to **Manage App → Advanced settings → Secrets** and add:

```toml
OPENAI_API_KEY = "your_openai_key"
OPENAI_API_BASE = "https://api.openai.com/v1"

[OXFORD]
APP_ID  = "your_oxford_app_id"
APP_KEY = "your_oxford_app_key"
```

### Environment Variables

If running locally (or in another hosting environment), you can instead set environment variables directly. For Together.ai’s OpenAI-compatible endpoint, first register at [together.ai](https://docs.together.ai/docs/quickstart) to get:

```bash
export TOGETHER_API_KEY="your_together_api_key"
export TOGETHER_API_BASE="https://api.together.xyz/v1"
```
Note: TOGETHER_API_KEY is used as OPENAI_API_KEY and similarly TOGETHER_API_BASE is used as OPENAI_API_BASE.
Then, in your shell or CI/CD environment, map these to the variables your code expects:

```bash
export OPENAI_API_KEY="$TOGETHER_API_KEY"
export OPENAI_API_BASE="$TOGETHER_API_BASE"
export OXFORD_APP_ID="your_oxford_app_id"
export OXFORD_APP_KEY="your_oxford_app_key"
```

---

## Usage

### CLI Mode

```bash
python main.py
```

* Enter your question when prompted.
* Outputs:

  1. **Tool/Agent Branch Used**
  2. **Retrieved Context Snippets** (if any)
  3. **Final Answer**

### Streamlit Mode

```bash
streamlit run app.py
```

* **Sidebar**: browse raw documents in `docs/`.
* **Main Panel**: ask questions and view tool branch, context snippets, and answer.

---

## Project Structure

```
knowledge-assistant/
├── app.py                  # Streamlit interface
├── main.py                 # CLI entrypoint
├── requirements.txt        # Hosting-ready dependencies
├── config.py               # Paths, model & index constants
├── utils.py                # Document loading & chunking utilities
├── src/
│   ├── llm.py              # Agent logic: CALCULATE, DEFINE, RAG
│   └── rag.py              # FAISS index & retriever initialization
├── docs/                   # Raw .txt documents for ingestion
└── data/                   # Saved FAISS index
```

---

## Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature/xyz`)
3. Commit your changes (`git commit -m "feat: add xyz"`)
4. Push (`git push origin feature/xyz`)
5. Open a Pull Request
