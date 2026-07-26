# AI Research & Knowledge Assistant — Starter scaffold

This repository is a starter scaffold implementing the required backend pieces for the "AI Research & Knowledge Assistant" assignment. It provides a minimal, production-oriented project structure and a small FastAPI application with stubs and working implementations for document upload, basic PDF extraction, simple chunking, embedding generation (using sentence-transformers), an in-memory vector search (NearestNeighbors from scikit-learn), and a placeholder for TensorFlow classifier integration.

This scaffold is intended to get you started quickly. It contains:

- A FastAPI app (src/main.py) with REST endpoints for upload, listing documents, processing, semantic search, and simple QA over retrieved chunks.
- Utilities for PDF text extraction and chunking (src/utils.py).
- A simple persistence layout (data/ for metadata and vectors, models/ for saved TensorFlow model files).
- A requirements.txt with suggested packages.

What I added
- Basic working pipeline for upload -> extract -> chunk -> embed -> index (in-memory + persisted to disk via joblib).
- Endpoints you can extend to add advanced features: RAG with an LLM, FAISS-based indexing, advanced chunking, TF model training, conversation memory.

How to run (development)

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start the FastAPI server:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

3. Open docs at http://localhost:8000/docs to try the endpoints.

Files you should add / tasks to finish the assignment
- Put a trained TensorFlow model in models/classifier.h5 or SavedModel at models/saved_model/ and the app will attempt to load it for automatic classification.
- Add sample PDFs to test the upload and processing pipeline.
- Improve chunking (e.g., semantic-aware chunking, overlap strategy) and justify choices in README.
- Integrate a vector DB (FAISS, Milvus, Pinecone) for scale and persistence.
- Add conversation memory (Redis or database-backed session store).
- Add unit tests and CI workflow.

Notes
- This scaffold avoids adding heavy platform-specific binary deps (like faiss) to keep initial setup simple. Replace scikit-learn NearestNeighbors with FAISS if you need production-scale retrieval.

---

If you'd like, I can now:
- Expand the implementation to include FAISS-based indexing and an example Dockerfile and GitHub Actions workflow.
- Add a small sample dataset (3 PDFs) into the repo so you can test immediately.
- Implement the TensorFlow model training pipeline and add a sample trained model.

Tell me which of these you'd like me to do next and I'll continue.
