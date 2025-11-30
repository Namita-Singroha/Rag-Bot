import faiss
import pickle
from sentence_transformers import SentenceTransformer
from config import INDEX_DIR, EMBED_MODEL_NAME
from typing import List, Dict

sentence_model = None
index = None
texts = None
metadata = None

def initialize():
    global sentence_model, index, texts, metadata
    
    if sentence_model is not None:
        return
    
    print("🚀 Loading model and index...")
    sentence_model = SentenceTransformer(EMBED_MODEL_NAME)
    
    if not (INDEX_DIR / "faiss.index").exists():
        print("📦 Creating new empty index...")
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        d = 384
        index = faiss.IndexFlatL2(d)
        texts = []
        metadata = []
        save_index()
    else:
        index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        with open(INDEX_DIR / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        with open(INDEX_DIR / "texts.pkl", "rb") as f:
            texts = pickle.load(f)
    
    print("✅ Loaded successfully!")

def retrieve_chunks(query: str, top_k: int = 4) -> List[Dict]:
    q_emb = sentence_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    
    chunks = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        chunks.append({
            "text": texts[idx],
            "source": metadata[idx].get("source", "unknown"),
            "chunk_id": metadata[idx].get("chunk_id", idx),
            "score": float(score)
        })
    return chunks

def add_to_index(filename: str, chunks: List[str]) -> int:
    global index, texts, metadata
    
    if not chunks:
        return 0
    
    embeddings = sentence_model.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    from datetime import datetime
    for chunk in chunks:
        texts.append(chunk)
        metadata.append({
            "source": filename,
            "chunk_id": len(texts) - 1,
            "upload_date": datetime.now().isoformat()
        })
    
    save_index()
    return len(chunks)

def save_index():
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "texts.pkl", "wb") as f:
        pickle.dump(texts, f)
    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

def get_stats():
    docs = {}
    for meta in metadata:
        source = meta.get("source", "unknown")
        if source not in docs:
            docs[source] = {"name": source, "chunks": 0, "upload_date": meta.get("upload_date", "unknown")}
        docs[source]["chunks"] += 1
    
    return {
        "total_documents": len(docs),
        "total_chunks": len(texts),
        "documents": list(docs.values())
    }