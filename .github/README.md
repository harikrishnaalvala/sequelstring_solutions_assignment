# AI Research & Knowledge Assistant

A FastAPI-based backend for an AI-powered document processing and semantic search system. This application enables users to upload PDF documents, extract and chunk text, generate embeddings, and perform semantic search with QA capabilities.

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Future Enhancements](#future-enhancements)

## ✨ Features

- **PDF Upload & Processing**: Upload PDF documents and automatically extract text
- **Text Chunking**: Intelligent text chunking with configurable overlap for better context preservation
- **Semantic Search**: Search documents using natural language queries powered by sentence transformers
- **Embedding Generation**: Generate embeddings using pre-trained sentence transformer models
- **Question Answering**: Basic RAG-style question answering with retrieved document context
- **Document Management**: Track uploaded documents with metadata (name, upload time, status, page count)
- **Persistent Storage**: Embeddings and metadata are persisted to disk via joblib and JSON

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Web Framework** | FastAPI |
| **ASGI Server** | Uvicorn |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **PDF Processing** | PyPDF2 |
| **Search** | Scikit-learn NearestNeighbors |
| **ML Framework** | TensorFlow |
| **Serialization** | Joblib, JSON |

## 📁 Project Structure

```
sequelstring_solutions_assignment/
├── src/
│   ├── main.py              # FastAPI application and endpoints
│   ├── utils.py             # Utility functions (PDF extraction, chunking, embeddings, search)
│   └── models/              # Directory for trained TensorFlow models
├── data/
│   ├── docs/                # Uploaded PDF files
│   ├── documents.json       # Document metadata
│   └── vectors.joblib       # Persisted embeddings and metadata
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd sequelstring_solutions_assignment
   ```

2. **Create a Python virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **On macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```
   - **On Windows**:
     ```bash
     .venv\Scripts\activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Quick Start

1. **Start the FastAPI development server**:
   ```bash
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open the interactive API documentation**:
   - Navigate to `http://localhost:8000/docs` in your browser
   - Swagger UI will show all available endpoints with interactive testing

3. **Test the API**:
   - Use the Swagger UI to upload a PDF, process it, and run searches

## 📡 API Endpoints

### 1. Upload PDF
**POST** `/upload`

Upload a PDF document to the system.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Parameter**: `file` (PDF file)

**Response** (201 Created):
```json
{
  "document_id": "uuid-string",
  "message": "Uploaded"
}
```

---

### 2. List Documents
**GET** `/documents`

Retrieve metadata for all uploaded documents.

**Response** (200 OK):
```json
[
  {
    "id": "doc-uuid",
    "name": "example.pdf",
    "path": "data/docs/doc-uuid.pdf",
    "upload_ts": "2026-07-26T10:30:45.123456Z",
    "pages": null,
    "chunks": 0,
    "status": "uploaded"
  }
]
```

**Statuses**:
- `uploaded`: Document uploaded but not yet processed
- `processed`: Text extracted, chunked, and embeddings generated

---

### 3. Process Document
**POST** `/process/{doc_id}`

Extract text, chunk, and generate embeddings for a document.

**Path Parameters**:
- `doc_id`: The UUID of the document to process

**Response** (200 OK):
```json
{
  "document_id": "doc-uuid",
  "chunks": 42
}
```

---

### 4. Semantic Search
**POST** `/search`

Search across all processed documents using semantic similarity.

**Request Body**:
```json
{
  "query": "What is machine learning?",
  "top_k": 5
}
```

**Response** (200 OK):
```json
{
  "results": [
    {
      "score": 0.87,
      "metadata": {
        "document_id": "doc-uuid",
        "page": 1,
        "text": "Machine learning is a subset of artificial intelligence..."
      }
    }
  ]
}
```

**Parameters**:
- `query` (required): Natural language search query
- `top_k` (optional, default: 5): Number of top results to return

---

### 5. Question Answering (QA)
**POST** `/qa`

Retrieve relevant document chunks for a question (RAG-style). Includes placeholder for LLM integration.

**Request Body**:
```json
{
  "query": "How does semantic search work?",
  "top_k": 5
}
```

**Response** (200 OK):
```json
{
  "answer": "[LLM integration required to generate a final answer. Use retrieved context to prompt your model.]",
  "retrieved": [
    {
      "score": 0.92,
      "metadata": {
        "document_id": "doc-uuid",
        "page": 2,
        "text": "Semantic search uses embeddings to understand..."
      }
    }
  ]
}
```

## 📝 Usage Examples

### Complete Workflow

```bash
# 1. Start the server
uvicorn src.main:app --reload

# 2. Upload a PDF (in another terminal)
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload
# Response: {"document_id": "abc-123-def", "message": "Uploaded"}

# 3. Process the document
curl -X POST http://localhost:8000/process/abc-123-def
# Response: {"document_id": "abc-123-def", "chunks": 42}

# 4. Search documents
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?", "top_k": 3}'

# 5. Ask a question
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain embeddings", "top_k": 3}'

# 6. List all documents
curl http://localhost:8000/documents
```

## ⚙️ Configuration

### Text Chunking

The text chunking strategy is defined in `src/utils.py`:

```python
chunk_text(text, chunk_size=1000, overlap=200)
```

- **chunk_size**: Number of characters per chunk (default: 1000)
- **overlap**: Number of overlapping characters between chunks (default: 200)

Adjust these values based on:
- **Large documents**: Increase `chunk_size` for efficiency
- **Context preservation**: Increase `overlap` to maintain context across chunks
- **Fine-grained search**: Decrease both for more granular results

### Embedding Model

The embedding model can be changed in `src/utils.py`:

```python
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
```

Alternatives:
- `all-MiniLM-L6-v2`: Fast, lightweight (384-dim) - recommended for quick setup
- `all-mpnet-base-v2`: High quality (768-dim) - slower but more accurate
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`: Multilingual support

## 🔮 Future Enhancements

### High Priority

- [ ] **Vector Database Integration**: Replace in-memory NearestNeighbors with FAISS, Milvus, or Pinecone for scalability
- [ ] **LLM Integration**: Connect to OpenAI, Anthropic, or local models for actual QA answer generation
- [ ] **Unit Tests**: Add comprehensive test suite with pytest
- [ ] **CI/CD Pipeline**: Add GitHub Actions workflow for automated testing and deployment
- [ ] **Conversation Memory**: Implement Redis or database-backed session store for multi-turn conversations

### Medium Priority

- [ ] **Advanced Chunking**: Implement semantic-aware chunking based on sentence boundaries and paragraph structure
- [ ] **TensorFlow Model Training**: Add pipeline for training document classifiers
- [ ] **API Authentication**: Add JWT or OAuth2 for secure endpoint access
- [ ] **Async Processing**: Use Celery for background document processing
- [ ] **Docker Support**: Add Dockerfile and docker-compose.yml for containerized deployment

### Lower Priority

- [ ] **Web UI**: Build frontend with React or Vue.js
- [ ] **GraphQL API**: Alternative to REST endpoints
- [ ] **Monitoring & Logging**: Add structured logging and performance monitoring
- [ ] **Batch Processing**: Support bulk document uploads and processing
- [ ] **Export Features**: Export search results and generated answers in multiple formats

## 📚 Resource Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [PyPDF2 Documentation](https://pypdf2.readthedocs.io/)
- [Scikit-learn NearestNeighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [Uvicorn Server](https://www.uvicorn.org/)

## 📄 License

This project is provided as-is for educational and assignment purposes.

---
