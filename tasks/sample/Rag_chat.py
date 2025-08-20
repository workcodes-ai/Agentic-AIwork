import os
import requests
import fitz
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from typing import List

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def download_pdf(url : str, save_path : str = "f1_technical.pdf"):
    response = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(response.content)
    return save_path

def extract_from_pdf(pdf_path : str) -> List[str]:
    doc = fitz.open(pdf_path)
    chunks = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            chunks.append(text.strip())
    return chunks


def build_faiss_index(chunks: List[str], model_name: str = 'all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy = True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, chunks, model


def retreive_context(query: str, index, chunks, model, k: int = 3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def generate_answer(query: str, context: List[str], model_name: str = "llama3-70b-8192"):
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{''.join(context)}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def chat_loop(index, chunks, model): 
    print("\n Ask Questions based on the PDf context(type 'exit' to quit):\n")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        context = retreive_context(query, index, chunks, model)
        answer = generate_answer(query, context)
        print("Bot: ", answer, '\n')


if __name__ == "__main__":
    
    pdf_url = "https://www.fia.com/sites/default/files/fia_2026_formula_1_technical_regulations_issue_8_-_2024-06-24.pdf"  
    pdf_path = download_pdf(pdf_url, save_path="latest_doc.pdf")

    text_chunks = extract_from_pdf(pdf_path)
    print(f"Extracted {len(text_chunks)} chunks from new PDF.")

    index, chunks, model = build_faiss_index(text_chunks)
    chat_loop(index, chunks, model)
