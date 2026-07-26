import PyPDF2
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib
from pathlib import Path

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_MODEL = None

def _get_model():
    global EMBED_MODEL
    if EMBED_MODEL is None:
        EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return EMBED_MODEL


def extract_text_from_pdf(path: str) -> Tuple[str, int]:
    """Extract text from PDF and return combined text and number of pages."""
    reader = PyPDF2.PdfReader(path)
    pages = len(reader.pages)
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts), pages


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple sliding-window chunking by characters. Replace with smarter token-based chunking for production."""
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == L:
            break
        start = end - overlap
    return chunks


def get_embeddings(texts: List[str]):
    model = _get_model()
    emb = model.encode(texts, show_progress_bar=False)
    return [e.tolist() for e in emb]


def save_index(path: str, store: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(store, path)


def load_index(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    return joblib.load(path)


def search_index(store: dict, query: str, top_k: int = 5):
    if not store or not store.get("embeddings"):
        return []
    model = _get_model()
    q_emb = model.encode([query])[0]
    X = np.array(store["embeddings"])
    nbrs = NearestNeighbors(n_neighbors=min(top_k, len(X)), metric="cosine")
    nbrs.fit(X)
    dists, idxs = nbrs.kneighbors([q_emb])
    out = []
    for dist, i in zip(dists[0], idxs[0]):
        meta = store["metadatas"][i]
        out.append({"score": float(1 - dist), "metadata": meta})
    return out
