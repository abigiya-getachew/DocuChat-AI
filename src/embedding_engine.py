import ollama
from config import Config

def get_embedding(text):
    response = ollama.embeddings(
        model=Config.EMBEDDING_MODEL,
        prompt=text
    )
    return response["embedding"]

def embed_documents(chunks):
    embedded = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, 1):
        if idx % 50 == 0 or idx == total:
            print(f"Embedding progress: {idx}/{total} chunks...")
        chunk["embedding"] = get_embedding(chunk["text"])
        embedded.append(chunk)
    return embedded