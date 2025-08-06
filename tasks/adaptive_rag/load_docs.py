import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.docstore.document import Document

def load_docs_from_folder(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())

        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
    return docs