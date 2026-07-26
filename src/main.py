from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid
import shutil
import json
from typing import List

from src.utils import extract_text_from_pdf, chunk_text, get_embeddings, save_index, load_index, search_index

DATA_DIR = Path("data")
DOCS_DIR = DATA_DIR / "docs"
METADATA_FILE = DATA_DIR / "documents.json"
VECTORS_FILE = DATA_DIR / "vectors.joblib"

DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
if not METADATA_FILE.exists():
    METADATA_FILE.write_text("[]")

app = FastAPI(title="AI Research & Knowledge Assistant - Starter")

# Load or initialize index
index_store = load_index(VECTORS_FILE)  # returns dict with 'embeddings' and 'metadatas' or empty

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported in this starter scaffold")
    doc_id = str(uuid.uuid4())
    dest = DOCS_DIR / f"{doc_id}.pdf"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    # Record metadata
    docs = json.loads(METADATA_FILE.read_text())
    meta = {
        "id": doc_id,
        "name": file.filename,
        "path": str(dest),
        "upload_ts": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "pages": None,
        "chunks": 0,
        "status": "uploaded"
    }
    docs.append(meta)
    METADATA_FILE.write_text(json.dumps(docs, indent=2))
    return JSONResponse(status_code=201, content={"document_id": doc_id, "message": "Uploaded"})

@app.get("/documents")
def list_documents():
    return JSONResponse(content=json.loads(METADATA_FILE.read_text()))

@app.post("/process/{doc_id}")
def process_document(doc_id: str):
    docs = json.loads(METADATA_FILE.read_text())
    match = next((d for d in docs if d["id"] == doc_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Document not found")
    # Extract text
    text, pages = extract_text_from_pdf(match["path"])  # returns combined text and page count
    # Chunk
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    # Create embeddings
    embeddings = get_embeddings(chunks)
    # Save into index store (append)
    for i, (ch, emb) in enumerate(zip(chunks, embeddings)):
        meta = {"document_id": doc_id, "page": i+1, "text": ch}
        index_store.setdefault("embeddings", []).append(emb)
        index_store.setdefault("metadatas", []).append(meta)
    save_index(VECTORS_FILE, index_store)
    # Update metadata
    match["pages"] = pages
    match["chunks"] = len(chunks)
    match["status"] = "processed"
    METADATA_FILE.write_text(json.dumps(docs, indent=2))
    return {"document_id": doc_id, "chunks": len(chunks)}

@app.post("/search")
def semantic_search(query: dict):
    """Request body: {"query": "...", "top_k": 5}"""
    q = query.get("query")
    top_k = int(query.get("top_k", 5))
    if not q:
        raise HTTPException(status_code=400, detail="Missing query")
    if not index_store.get("embeddings"):
        return {"results": []}
    results = search_index(index_store, q, top_k=top_k)
    return {"results": results}

@app.post("/qa")
def qa(query: dict):
    """Simple RAG-like QA: retrieve top chunks and return as context.
    Request body: {"query":"...", "top_k": 5}
    """
    q = query.get("query")
    top_k = int(query.get("top_k", 5))
    if not q:
        raise HTTPException(status_code=400, detail="Missing query")
    results = search_index(index_store, q, top_k=top_k)
    # NOTE: This scaffold does NOT call a hosted LLM. It returns the retrieved context and a placeholder answer.
    answer = "[LLM integration required to generate a final answer. Use retrieved context to prompt your model.]"
    return {"answer": answer, "retrieved": results}
