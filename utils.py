# creating functions to read text and pdf files


from pathlib import Path
from pypdf import PdfReader

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text or ""
        pages.append(text)
    return "/n".join(pages)

def load_document(path:str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".txt":
        return read_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start+ chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size-overlap
    
    return chunks
