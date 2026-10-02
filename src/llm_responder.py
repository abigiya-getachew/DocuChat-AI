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

    # 1. Use Groq Cloud API if key is provided (ideal for Render Free Tier)
    if Config.GROQ_API_KEY:
        try:
            model = Config.DEFAULT_MODEL
            # Default to qwen/qwen3.8-27b or openai/gpt-oss-120b if llama3 is specified for Groq
            if "llama" in model.lower() and "prompt-guard" not in model.lower():
                model = "qwen/qwen3.8-27b"
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {Config.GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 1024
                },
                timeout=60
            )
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                msg = data["choices"][0]["message"]
                content = msg.get("content") or msg.get("reasoning") or ""
                if content.strip():
                    return content.strip()
            elif "error" in data:
                err_msg = data["error"].get("message", str(data["error"]))
                print(f"Groq API Error: {err_msg}")
        except Exception as e:
            if not Config.OLLAMA_HOST or "localhost" not in Config.OLLAMA_HOST:
                raise e
            print(f"Groq failed, trying Ollama: {e}")

    # 2. Local / Remote Ollama execution
    headers = {}
    if Config.OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {Config.OLLAMA_API_KEY}"

    host = Config.OLLAMA_HOST
    try:
        response = requests.post(
            f"{host}/api/chat",
            headers=headers,
            json={
                "model": Config.DEFAULT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=120
        )
        data = response.json()
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        elif "error" in data:
            # If remote returned an error, attempt fallback to local Ollama
            if "localhost" not in host and "127.0.0.1" not in host:
                local_resp = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "llama3.2",
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False
                    },
                    timeout=120
                )
                local_data = local_resp.json()
                if "message" in local_data and "content" in local_data["message"]:
                    return local_data["message"]["content"]
            raise Exception(data["error"])
        else:
            raise Exception(f"Unexpected LLM response: {data}")
    except Exception as e:
        if "localhost" not in host and "127.0.0.1" not in host:
            try:
                local_resp = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "llama3.2",
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False
                    },
                    timeout=120
                )
                local_data = local_resp.json()
                if "message" in local_data and "content" in local_data["message"]:
                    return local_data["message"]["content"]
            except Exception:
                pass
        raise e