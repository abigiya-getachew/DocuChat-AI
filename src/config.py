import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2:1b")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    DATA_FOLDER = os.getenv("DATA_FOLDER", "./data/uploaded_docs")
    CHROMA_PATH = "./chroma_db"