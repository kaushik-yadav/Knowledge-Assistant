import os
import re

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

docs = {}
# taking all the text files present in docs
txt_files = [file for file in os.listdir("docs/") if file.endswith(".txt")]

for file in txt_files:
    with open(os.path.join("docs", file)) as f:
        docs[file.replace(".txt", "")] = f.read()


# chunking each document like {source : 'doc category', content : 'the content in category'}
def get_formatted_data(filename):
    with open(filename, "r") as f:
        data = f.read()
        source = filename.split(".")[0].split("/")[1]

        # splitting the product specs data into seperate product wise data
        if source == "products_specs":
            split_text = re.split("(?=Product Name)", data)[1:]

        # splitting the company faqs into pairs of Q and A
        # for company overview splitting it paragraph wise
        else:
            split_text = re.split("\n\n", data)
        return [{"source": source, "content": x} for x in split_text]


chunked_data = []

# storing the chunked data of each doc
for file in txt_files:
    chunked_data.extend(get_formatted_data(f"docs/{file}"))

# converting our chunks in documents

documents = []

for chunk in chunked_data:
    doc = Document(page_content=chunk["content"], metadata={"source": chunk["source"]})
    documents.append(doc)

# embedding
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# this embeds the docs and builds a FAISS index
vector_store = FAISS.from_documents(documents, embedding_model)

# save the index in docs/
vector_store.save_local("data/documents_index")

# loading the model
vector_store = FAISS.load_local(
    "data/documents_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,
)

# initializing the retriever to retrieve the top k relevant documents with a nearest neighbors search (fetching up to 50 candidates)
retriever = VectorStoreRetriever(
    vectorstore=vector_store, search_kwargs={"k": 5, "fetch_k": 50}
)

# retrieving the top k documents
top_k_queries = retriever.get_relevant_documents(
    "How does the policy of returning works in this company?"
)
