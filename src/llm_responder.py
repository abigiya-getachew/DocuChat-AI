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

import re as _re

def _clean_response(text: str) -> str:
    """Strip <think>...</think> blocks that some models (e.g. Qwen) emit."""
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL)
    return text.strip()

def answer_query(user_question):
    query_embedding = get_embedding(user_question)
    relevant_docs = search_relevant(query_embedding)
    context = build_context(relevant_docs)

    system_message = """You are DocuChat AI, a precise document assistant. Answer questions using ONLY the provided document context.

STRICT FORMATTING RULES:
1. Never use markdown headings (#, ##, ###). Use **bold labels** instead.
2. For lists, use numbered steps (1. 2. 3.) or bullet dashes (- item).
3. Cite every fact as (Source: filename, Page N) inline after the sentence.
4. Keep your answer focused and concise. No padding or filler sentences.
5. If the answer is not in the context, respond only with: "I cannot find that information in the provided documents."
6. Do not repeat the question back."""

    user_message = f"""--- Document Context ---
{context}

--- Question ---
{user_question}"""


    # 1. Use Groq Cloud API if key is provided (ideal for Render Free Tier)
    if Config.GROQ_API_KEY:
        try:
            model = Config.DEFAULT_MODEL
            # Default to qwen/qwen3.8-27b if llama3 model name is set locally
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
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1024
                },
                timeout=60
            )
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                msg = data["choices"][0]["message"]
                content = msg.get("content") or msg.get("reasoning") or ""
                content = _clean_response(content)
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
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
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
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message}
                        ],
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
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message}
                        ],
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