import os
from pathlib import Path

import PyPDF2
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Paths ---
PDF_PATH = Path("mybuild_.pdf")       
FAISS_DIR = Path("faiss_index")       


def _load_pdf_documents(pdf_path: Path):
    """Read pages from PDF and turn them into LangChain Documents."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    docs = []
    with pdf_path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf_path.name, "page": i},
                )
            )
    return docs


def _build_or_load_faiss_store():
    """Build FAISS index from PDF if missing, otherwise load from disk."""
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # If FAISS index exists → load it
    if FAISS_DIR.exists() and any(FAISS_DIR.iterdir()):
        print("🔁 Loading existing FAISS index from", FAISS_DIR)
        store = FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True, 
        )
        return store

    # Otherwise → build new index
    print(" Building FAISS index from", PDF_PATH)
    docs = _load_pdf_documents(PDF_PATH)
    if not docs:
        raise RuntimeError(f"No text extracted from: {PDF_PATH}")

    store = FAISS.from_documents(docs, embeddings)

    # Save for later reuse
    store.save_local(str(FAISS_DIR))
    print(f" FAISS index created with {len(docs)} chunks.")
    return store


# Build / load at import time
_faiss_store = _build_or_load_faiss_store()

# Create retriever
_retriever = _faiss_store.as_retriever(search_kwargs={"k": 5})


def get_retriever():
    """Used by app.py to retrieve documents."""
    return _retriever
