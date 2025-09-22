from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from contextlib import asynccontextmanager
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model on startup
    global model
    try:
        from mxbai_rerank import MxbaiRerankV2
        logger.info("Loading mxbai-rerank model...")
        model = MxbaiRerankV2("mixedbread-ai/mxbai-rerank-base-v2")
        logger.info("Model loaded successfully!")
        yield
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    finally:
        # Clean up resources
        model = None

app = FastAPI(
    title="MxBai Reranker Service",
    description="A FastAPI service for document reranking using mxbai-rerank model",
    version="1.0.0",
    lifespan=lifespan
)

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: Optional[int] = None
    return_documents: Optional[bool] = True

class RerankResponse(BaseModel):
    results: List[dict]
    total_documents: int
    query: str

@app.get("/")
async def root():
    return {"message": "MxBai Reranker Service is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "mixedbread-ai/mxbai-rerank-base-v2"}

@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """
    Rerank documents based on query relevance
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Reranking {len(request.documents)} documents for query: {request.query[:100]}...")
        
        # Perform reranking
        results = model.rank(
            query=request.query,
            documents=request.documents,
            return_documents=request.return_documents,
            top_k=request.top_k
        )
        
        # Convert RankResult objects to dictionaries
        formatted_results = []
        for result in results:
            formatted_results.append({
                "index": result.index,
                "score": result.score,
                "document": result.document if hasattr(result, 'document') else None
            })
        
        logger.info(f"Reranking completed. Returned {len(formatted_results)} results.")
        
        return RerankResponse(
            results=formatted_results,
            total_documents=len(request.documents),
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")

@app.post("/rerank/simple")
async def rerank_simple(request: RerankRequest):
    """
    Simple rerank endpoint that returns only scores and indices
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = model.rank(
            query=request.query,
            documents=request.documents,
            return_documents=False,
            top_k=request.top_k
        )
        
        # Convert RankResult objects to simple format
        return {
            "scores": [result.score for result in results],
            "indices": [result.index for result in results],
            "query": request.query
        }
        
    except Exception as e:
        logger.error(f"Error during simple reranking: {e}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
