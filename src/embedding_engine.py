import os
import ollama
from config import Config

_st_model = None

def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _st_model

# Ensure embeddings connect to local Ollama (api.ollama.com does not support /api/embeddings)
host = Config.OLLAMA_HOST
if "api.ollama.com" in host or not host:
    host = "http://localhost:11434"

client = ollama.Client(host=host)

def get_embedding(text):
    # Use sentence-transformers if on Render or configured for MiniLM
    if os.getenv("RENDER") or "MiniLM" in Config.EMBEDDING_MODEL or os.getenv("GROQ_API_KEY"):
        model = _get_st_model()
        return model.encode(text).tolist()

    # Otherwise use Ollama
    try:
        response = client.embeddings(
            model=Config.EMBEDDING_MODEL,
            prompt=text
        )
        return response["embedding"]
    except Exception:
        # Fallback to SentenceTransformer in-process
        model = _get_st_model()
        return model.encode(text).tolist()

def embed_documents(chunks):
    embedded = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, 1):
        if idx % 50 == 0 or idx == total:
            print(f"Embedding progress: {idx}/{total} chunks...")
        chunk["embedding"] = get_embedding(chunk["text"])
        embedded.append(chunk)
    return embedded