import os

from dotenv import load_dotenv

load_dotenv()

DOCS_PATH = "docs/"
INDEX_PATH = "data/documents_index/"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OXFORD_APP_ID = os.getenv("OXFORD_APP_ID")
OXFORD_APP_KEY = os.getenv("OXFORD_APP_KEY")
