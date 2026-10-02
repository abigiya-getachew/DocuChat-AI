import sys
sys.path.append("./src")

from src.document_loader import process_file
from src.embedding_engine import embed_documents
from src.vector_store import add_documents
from src.llm_responder import answer_query
from src.config import Config
import os

print("=== DocuChat AI ===")
files = os.listdir(Config.DATA_FOLDER)
pdf_files = [f for f in files if f.lower().endswith(".pdf")]

if not pdf_files:
    print(f"⚠️  Put your PDF files in: {Config.DATA_FOLDER}")
    sys.exit(0)

print(f"Found {len(pdf_files)} document(s): {pdf_files}")

for fname in pdf_files:
    print(f"Processing: {fname}...")
    chunks = process_file(fname)
    embedded = embed_documents(chunks)
    add_documents(embedded)
    print(f"✅ {fname} indexed")

print("\nReady! Ask questions or type 'exit' to quit")
while True:
    q = input("\nYour question: ")
    if q.lower() in ["exit", "quit"]:
        break
    print("Thinking...")
    ans = answer_query(q)
    print("\nAnswer:\n", ans)