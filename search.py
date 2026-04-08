import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_PATH = "index/faiss.index"
META_PATH = "index/metadata.json"

def search(query: str, top_k: int = 5):
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(FAISS_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx==-1:
            continue
        results.append(
            {
                "score": float(score),
                "source": metadata[idx]["source"],
                "chunk_id": metadata[idx]["chunk_id"],
                "text": metadata[idx]["text"]
            }
        )
    return results

if __name__ == "__main__":
    query = input("Enter search query: ")
    results = search(query, top_k=5)
    
    for i,r in enumerate(results, 1):
        print(f"\n Result #{i}")
        print(f"Source: {r['source']}")
        print(f"Chunk ID: {r['chunk_id']}")
        print(f"Similarity score: {r['score']:.4f}")
        print(f"text: {r['text'][:500]}...")