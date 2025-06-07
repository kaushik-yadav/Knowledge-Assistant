from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader

from config import EMBED_MODEL_NAME, INDEX_PATH
from utils import get_documents


def extract_text_from_pdf(uploaded_pdf) -> str:
    pdf_reader = PdfReader(uploaded_pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
        print(page.extract_text())
    return text

def build_retriever_from_uploaded_text(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store.as_retriever()


# load and chunk documents once
documents = get_documents()

# loading the embedding model
embedding_model = HuggingFaceEmbeddings(model_name=f"{EMBED_MODEL_NAME}")

# this embeds the docs and builds a FAISS index
vector_store = FAISS.from_documents(documents, embedding_model)

# save the index in data/
vector_store.save_local(INDEX_PATH)

# loading the FAISS index
vector_store = FAISS.load_local(
    INDEX_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,
)

# initializing the retriever to retrieve the top k relevant documents with a nearest neighbors search (fetching up to 70 candidates)
retriever = VectorStoreRetriever(
    vectorstore=vector_store, search_kwargs={"k": 3, "fetch_k": 70}
)

def start_rag():
    return retriever