import chromadb
from config import Config

client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
_coll_name = f"docuchat_{Config.EMBEDDING_MODEL.replace('-', '_').replace('/', '_')}"
collection = client.get_or_create_collection(name=_coll_name)

def add_documents(embedded_chunks, batch_size=100):
    for i in range(0, len(embedded_chunks), batch_size):
        batch = embedded_chunks[i:i + batch_size]
        collection.upsert(
            ids=[f"{c['source']}_p{c['page']}_{i + idx}" for idx, c in enumerate(batch)],
            embeddings=[c["embedding"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "source": c["source"],
                "page": c["page"]
            } for c in batch]
        )

def search_relevant(query_embedding, n_results=5):
    if collection.count() == 0:
        return []
    
    count = min(n_results, collection.count())
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=count
    )
    docs = []
    if results and results.get("documents") and results["documents"][0]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append({
                "source": meta["source"],
                "page": meta["page"],
                "text": doc
            })
    return docs

def remove_documents_by_source(source_filename):
    try:
        collection.delete(where={"source": source_filename})
        return True
    except Exception as e:
        print(f"Error removing document {source_filename} from Chroma: {e}")
        return False

def clear_all_documents():
    try:
        client.delete_collection("docuchat_docs")
        global collection
        collection = client.get_or_create_collection(name="docuchat_docs")
        return True
    except Exception as e:
        print(f"Error clearing Chroma collection: {e}")
        return False