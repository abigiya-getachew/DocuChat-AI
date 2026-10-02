import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile" if os.getenv("GROQ_API_KEY") else "llama3.2")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2" if os.getenv("RENDER") or os.getenv("GROQ_API_KEY") else "nomic-embed-text")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    DATA_FOLDER = os.getenv("DATA_FOLDER", "./data/uploaded_docs")
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")