# generate embeddings

import os
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from utils import load_document, chunk_text

MODEL_NAME = "all-MiniLLM-L6-v2"
DOCS_DIR = "data/docs"
INDEX_DIR = "index"
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "metadata.json")

def build_index():
    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []
    metadata = []

    for filename in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, filename)
        if not os.path.isfile(path):
            continue # if file doesnt exist in path then skip

        text = load_document(path)
        chunks = chunk_text(path)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)    
            metadata.append(
                {
                    "source": filename,
                    "chunk_id": i,
                    "text": chunk
                }
            )

    embeddings = model.encode(
        all_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    dimension = embeddings.shape[1]

    #cosine similarity via inner product on normalized vectors
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(all_chunks)} chunks")
    print(f"Embedding dimension: {dimension}")

if __name__=="__main__":
    build_index()

    
# Why normalize embeddings?

# Cosine similarity compares vectors by angle rather than raw length. A common trick is:

# normalize vectors to unit length
# use inner product search

# That makes inner product behave like cosine similarity.

# FAISS supports different index types and metrics, and its docs explain that vectors 
# are stored with fixed dimensionality in float32 arrays. 
# Sentence Transformers also documents semantic search workflows built on embeddings 
# and similarity search.