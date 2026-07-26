import PyPDF2
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib
from pathlib import Path

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_TOKENIZER = None
EMBED_MODEL = None

def _get_model():
    """Load the embedding model and tokenizer lazily."""
    global EMBED_TOKENIZER, EMBED_MODEL
    if EMBED_MODEL is None:
        EMBED_TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        EMBED_MODEL = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        # Set to eval mode for inference only
        EMBED_MODEL.eval()
    return EMBED_TOKENIZER, EMBED_MODEL


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
    """Generate embeddings using transformers + mean pooling."""
    tokenizer, model = _get_model()
    
    # Tokenize with padding and truncation
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Generate embeddings without gradient computation
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling: average all token embeddings
    attention_mask = inputs["attention_mask"]
    last_hidden_state = outputs.last_hidden_state
    
    # Expand attention mask for broadcasting
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    
    # Avoid division by zero
    mean_embeddings = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
    
    # Convert to list of lists
    return [emb.cpu().numpy().tolist() for emb in mean_embeddings]


def save_index(path: str, store: dict):
    """Save embeddings and metadata to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(store, path)


def load_index(path: str):
    """Load embeddings and metadata from disk."""
    p = Path(path)
    if not p.exists():
        return {}
    return joblib.load(path)


def search_index(store: dict, query: str, top_k: int = 5):
    """Search for similar documents using cosine distance."""
    if not store or not store.get("embeddings"):
        return []
    
    tokenizer, model = _get_model()
    
    # Encode query
    q_inputs = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        q_outputs = model(**q_inputs)
    
    # Mean pooling for query
    q_attention_mask = q_inputs["attention_mask"]
    q_last_hidden_state = q_outputs.last_hidden_state
    q_input_mask_expanded = q_attention_mask.unsqueeze(-1).expand(q_last_hidden_state.size()).float()
    q_sum_embeddings = torch.sum(q_last_hidden_state * q_input_mask_expanded, 1)
    q_sum_mask = q_input_mask_expanded.sum(1)
    q_emb = (q_sum_embeddings / torch.clamp(q_sum_mask, min=1e-9)).cpu().numpy()[0]
    
    # Search using nearest neighbors
    X = np.array(store["embeddings"])
    nbrs = NearestNeighbors(n_neighbors=min(top_k, len(X)), metric="cosine")
    nbrs.fit(X)
    dists, idxs = nbrs.kneighbors([q_emb])
    
    out = []
    for dist, i in zip(dists[0], idxs[0]):
        meta = store["metadatas"][i]
        out.append({"score": float(1 - dist), "metadata": meta})
    return out
