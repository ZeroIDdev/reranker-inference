# MxBai Reranker Service

A FastAPI-based service for document reranking using the mxbai-rerank model from Hugging Face.

## Features

- **Document Reranking**: Rerank documents based on query relevance using mixedbread-ai/mxbai-rerank-base-v2 model
- **FastAPI Framework**: Modern, fast web framework with automatic API documentation
- **Health Checks**: Built-in health monitoring endpoints
- **Flexible API**: Multiple endpoints for different use cases
- **Async Support**: Asynchronous processing for better performance

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Start the Server

```bash
python main.py
```

The server will start on `http://localhost:8001`

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Document Reranking
```bash
POST /rerank
```

**Request Body:**
```json
{
  "query": "Who wrote 'To Kill a Mockingbird'?",
  "documents": [
    "Document 1 text...",
    "Document 2 text..."
  ],
  "top_k": 3,
  "return_documents": true
}
```

**Response:**
```json
{
  "results": [
    {
      "score": 0.95,
      "index": 0,
      "document": "Document text..."
    }
  ],
  "total_documents": 6,
  "query": "Who wrote 'To Kill a Mockingbird'?"
}
```

#### Simple Reranking (Scores Only)
```bash
POST /rerank/simple
```

Returns only scores and indices without document text.

### Testing

Run the test script to verify the service is working:

```bash
python test_reranker.py
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## Integration Example

```python
import requests

def rerank_documents(query, documents, top_k=5):
    response = requests.post("http://localhost:8001/rerank", json={
        "query": query,
        "documents": documents,
        "top_k": top_k,
        "return_documents": True
    })
    return response.json()

# Usage
results = rerank_documents(
    query="What is machine learning?",
    documents=["Doc 1", "Doc 2", "Doc 3"],
    top_k=3
)
```

## Model Information

- **Model**: mixedbread-ai/mxbai-rerank-base-v2
- **Purpose**: Document reranking based on semantic similarity
- **Performance**: Optimized for accuracy and speed

## Configuration

The server runs on port 8001 by default. You can modify this in `main.py`:

```python
uvicorn.run("main:app", host="0.0.0.0", port=8001)
```
