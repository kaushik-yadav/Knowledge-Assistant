import os
import re

from langchain.schema import Document

from constants import DOCS_PATH


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


def get_documents():
    # taking all the text files present in docs
    txt_files = [file for file in os.listdir(DOCS_PATH) if file.endswith(".txt")]

    chunked_data = []

    # storing the chunked data of each doc
    for file in txt_files:
        joined_path = os.path.join(DOCS_PATH, file)
        chunked_data.extend(get_formatted_data(joined_path))

    # converting our chunks into documents
    documents = []

    for chunk in chunked_data:
        doc = Document(
            page_content=chunk["content"], metadata={"source": chunk["source"]}
        )
        documents.append(doc)

    return documents
