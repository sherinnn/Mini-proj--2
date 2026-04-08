# 📄 Semantic Search Engine for Documents

A simple end-to-end **Semantic Search Engine** built with Python using **embeddings + FAISS**.
This project demonstrates how raw text documents are converted into vectors and retrieved using similarity search, the core idea behind modern **RAG (Retrieval-Augmented Generation)** systems.

---

## 🚀 Project Overview

This system allows you to:

* Upload PDF or text documents
* Convert them into smaller chunks
* Generate embeddings (vector representations)
* Store them in a vector database (FAISS)
* Perform semantic search using natural language queries

---

## 🧠 How It Works

```
Documents → Chunking → Embeddings → FAISS Index → Query → Top-K Results
```

### Pipeline

1. **Load documents** (`.pdf`, `.txt`)
2. **Extract text**
3. **Split into chunks**
4. **Convert chunks into embeddings**
5. **Store vectors in FAISS**
6. **Search using cosine similarity**
7. **Return most relevant chunks**

---

## 🛠️ Tech Stack

* Python
* Sentence Transformers (`all-MiniLM-L6-v2`)
* FAISS (vector search)
* PyPDF (PDF text extraction)
* NumPy

---

## 📥 Add Documents

Place your files inside:

```
data/docs/
```

Supported formats:

* `.pdf`
* `.txt`

---

## 🔧 Build the Index

```bash
python ingest.py
```

This will:

* Extract text from documents
* Chunk the text
* Generate embeddings
* Store vectors in FAISS

---

## 🔍 Search

```bash
python search.py
```

Example:

```
Enter search query: what did voldemort do
```

Output:

```
Result #1
Source: document.pdf
Chunk ID: 12
Similarity Score: 0.65
Text: Voldemort...
```

---

## 📊 Key Concepts Learned

### 🔹 Text Chunking

Break large documents into smaller pieces for better retrieval.

### 🔹 Embeddings

Convert text into numerical vectors representing meaning.

### 🔹 Cosine Similarity

Measure how close two vectors are in semantic space.

### 🔹 Vector Databases (FAISS)

Efficiently store and search embeddings.

### 🔹 Retrieval (RAG Foundation)

Retrieve relevant context before using LLMs.
