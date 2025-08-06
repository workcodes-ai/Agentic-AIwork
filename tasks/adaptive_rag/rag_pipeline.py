from langchain_core.prompts import PromptTemplate

rag_prompt = PromptTemplate.from_template("""
You are an expert assistant. Use the following context to answer the question.
If the context is not helpful, answer it to the best of your ability.

Context:
{context}

Question:
{question}
""")

def adaptive_rag(query: str, vectorstore, embeddings, llm, threshold=0.75):
    # print("🔍 Performing similarity search (with scores)...")
    
    results = vectorstore.similarity_search_with_score(query, k=3)
    # print("📊 Raw Results:", results)

    if not results:
        # print("⚠️ No vector DB matches. Sending query directly to LLM...")
        response = llm.invoke(query)
        # print("🧠 Raw response object:", response)
        # print("🧠 Response content:", getattr(response, "content", "❌ No .content attribute"))
        return getattr(response, "content", "No response received"), True

    docs, scores = zip(*results)
    # print("📊 Similarity scores:", scores)

    max_score = max(scores)
    if max_score < threshold:
        # print(f"⚠️ Score too low ({max_score:.2f} < {threshold}). Using LLM fallback...")
        response = llm.invoke(query)
        # print("🧠 Raw response object:", response)
        # print("🧠 Response content:", getattr(response, "content", "❌ No .content attribute"))
        return getattr(response, "content", "No response received"), True

    context = "\n".join([doc.page_content for doc in docs])
    prompt = rag_prompt.format(context=context, question=query)
    # print("📨 Prompt to LLM:\n", prompt)

    response = llm.invoke(prompt)
    # print("🧠 Raw response object:", response)
    # print("🧠 Response content:", getattr(response, "content", "❌ No .content attribute"))
    return getattr(response, "content", "No response received"), False


def add_to_db(answer: str, query: str, vectorstore, embeddings):
    from langchain.docstore.document import Document
    doc = Document(page_content=answer, metadata={"source": "llm-generated", "query":query})
    vectorstore.add_documents([doc])
