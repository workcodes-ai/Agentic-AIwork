import os
from dotenv import load_dotenv
load_dotenv()

from load_docs import load_docs_from_folder
from vector_db import build_vector
from rag_pipeline import adaptive_rag, add_to_db
from langchain_groq import ChatGroq



def main():
    folder_path = 'knowledge_base'
    docs = load_docs_from_folder(folder_path)
    vectorstore, embeddings = build_vector(docs)
    


    llm = ChatGroq(
        groq_api_key = os.getenv("GROQ_API_KEY"),
        model_name = "llama3-70b-8192",
        temperature=0.2
    )

    while True:
        query = input("\n Ask me something (or type exit): ")
        if query.lower() in ["exit", "quit"]:
            break
        
        pending_add = None
        
        

        try:
            print("🤖 Thinking...")
            answer, used_fallback = adaptive_rag(query, vectorstore, embeddings, llm)

            # ✅ Display first
            print("\n====================")
            print(f"🧾 Query   : {query}")
            print(f"💬 Response: {answer}")
            print(f"📦 Source  : {'LLM (fallback)' if used_fallback else 'VectorDB'}")
            print("====================\n")

            # ✅ Add to DB AFTER displaying
            if used_fallback:
                print("📥 Saving LLM-generated answer into vectorstore...")
                add_to_db(answer, query, vectorstore, embeddings)
                print(f"📚 [ADAPTIVE] Added new entry for query: '{query}'\n→ Answer: {answer[:100]}...")

        except Exception as e:
            print("❌ ERROR:", e)
        
        if pending_add is not None:
            ans, q = pending_add
            print("📥 Saving LLM-generated answer into vectorstore...")
            add_to_db(ans, q, vectorstore, embeddings)
            print(f"📚 [ADAPTIVE] Added new entry for query: '{q}'\n→ Answer: {ans[:100]}...")
            pending_add = None
    




if __name__ == "__main__":
    main()
