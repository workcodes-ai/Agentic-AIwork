from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def load_file(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file_path)
    elif ext == ".docx" or ".doc":
        from langchain.document_loaders import UnstructuredWordDocumentLoader
        loader - UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError("Unsupported file")
    
    return loader.load()

def chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 30)
    return splitter.split_documents(documents)

def vector_db(docs, save_path="knowledge_base"):
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(save_path)
    print(f"Vector DB saved to: {save_path}")



if __name__ == "__main__":
    file_path = "C:\Users\SMBXL GUEST\AgentiAi\tasks\f1_technical.pdf"

    print("Loading and parsing document")
    raw_docs = load_file(file_path)

    print("Creating Chunks")
    chunks1 = chunks(raw_docs)

    print("Creating Vector Db")
    vector_db(chunks1)