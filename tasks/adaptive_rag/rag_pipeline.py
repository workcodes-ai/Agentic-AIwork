from langchain_core.prompts import PromptTemplate

rag_prompt = PromptTemplate.from_template("""
You are an expert assistant. Use the following context to answer the question.
If the context is not helpful, answer it to the best of your ability.

Context:
{context}

Question:
{question}
""")

def adaptive_rag(query, vectorstore, embeddings, llm, k=3, similarity_threshold=0.7):
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    query_embedding = embeddings.embed_query(query)


    if not results_with_scores:
        response = llm.invoke(query)
        return getattr(response, "content", "No response received"), True  # fallback

    # Filter docs by threshold
    filtered_docs = [(doc, score) for doc, score in results_with_scores if score >= similarity_threshold]

    

    # Build context from filtered docs
    context = "\n".join([doc.page_content for doc, _ in filtered_docs])
    prompt = rag_prompt.format(context=context, question=query)

    response = llm.invoke(prompt)
    return getattr(response, "content", "No response received"), False  # used vectorstore



def add_to_db(answer: str, query: str, vectorstore, embeddings):
    from langchain.docstore.document import Document
    doc = Document(
        page_content=answer,
        metadata={"source": "llm-generated", "query": query}
    )
    vectorstore.add_documents([doc])

