from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBED_MODEL_NAME, INDEX_PATH
from utils import get_documents


def start_rag():

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

    # initializing the retriever to retrieve the top k relevant documents with a nearest neighbors search (fetching up to 50 candidates)
    retriever = VectorStoreRetriever(
        vectorstore=vector_store, search_kwargs={"k": 6, "fetch_k": 70}
    )

    return retriever
