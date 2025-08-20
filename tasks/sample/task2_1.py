from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

api_key = os.getenv("GROQ_API_KEY")

class_prompt = PromptTemplate.from_template("""
                                            You are an intelligent router for a multi-agent system.

Classify the user's message into one of the following:
- "knowledge" — for domain-specific questions like farming, crops, soil, etc.
- "weather" — for questions about current or forecasted weather.
- "ambiguous" — if the question is unclear or mixes multiple intents.

Respond with only one word: knowledge, weather, or ambiguous.

User Query:
{query}
""")
rag_prompt = PromptTemplate.from_template("""
Use the following context to answer the question accurately.

Context:
{context}

Question:
{query}

Answer:
""")

embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_knowledge_base", embedding_model, allow_dangerous_deserialization=True)


llm = ChatGroq(model = "gemma2-9b-it")

classifier = class_prompt | llm
def classify_node(query: str) -> str:
    result = classifier.invoke({"query": query})
    result_text = getattr(result, "content", str(result)).strip().lower()
    return result_text if result_text in ["knowledge", "weather", "ambiguous"] else "ambiguous"
    

def rag_node(query: str) -> str:
    docs = db.similarity_search(query, k = 3)
    context = "\n\n".join(doc.page_content for doc in docs)
    full_prompt = rag_prompt.format(context=context, query=query)
    response = llm.invoke(full_prompt)
    return getattr(response, "content", str(response)) 
    
    

if __name__ == "__main__":
    print("👋 Welcome! Ask me about farming or weather. Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        category = classify_node(query)

        if category == "knowledge":
            response = rag_node(query)
        
        else:
            response = "Can you please clarify your question?"

        print(f"🧠 Agent ({category}): {response}\n")