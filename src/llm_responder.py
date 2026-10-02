import requests
from config import Config
from embedding_engine import get_embedding
from vector_store import search_relevant

def build_context(relevant_docs):
    context_parts = []
    for doc in relevant_docs:
        context_parts.append(
            f"[Source: {doc['source']}, Page {doc['page']}]\n{doc['text']}"
        )
    return "\n\n".join(context_parts)

def answer_query(user_question):
    query_embedding = get_embedding(user_question)
    relevant_docs = search_relevant(query_embedding)
    context = build_context(relevant_docs)

    prompt = f"""Use ONLY the provided context below to answer.
Cite source and page number for every fact.
If answer is not in context, say "I cannot find that information in the provided documents."

--- Context ---
{context}
--- Question ---
{user_question}
"""

    response = requests.post(
        f"{Config.OLLAMA_HOST}/api/chat",
        headers={"Authorization": f"Bearer {Config.OLLAMA_API_KEY}"},
        json={
            "model": Config.DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )
    return response.json()["message"]["content"]