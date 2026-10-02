import os
from pypdf import PdfReader
from config import Config

def load_pdf(file_path):
    pages = []
    reader = PdfReader(file_path)
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages.append({"page": idx, "text": text})
    return pages

def split_text(text, chunk_size=Config.CHUNK_SIZE, overlap=Config.CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def process_file(filename):
    file_path = os.path.join(Config.DATA_FOLDER, filename)
    all_chunks = []
    pages = load_pdf(file_path)
    for page in pages:
        chunks = split_text(page["text"])
        for chunk in chunks:
            all_chunks.append({
                "source": filename,
                "page": page["page"],
                "text": chunk
            })
    return all_chunks