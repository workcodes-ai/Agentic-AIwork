from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_vector(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db, embeddings