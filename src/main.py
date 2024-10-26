# src/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from prometheus_client import start_http_server
import threading
from .nim_service import NIMInferenceService
from .config import settings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NIM Inference Service",
    description="High-performance inference service using NVIDIA Inference Microservices",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference service
inference_service = NIMInferenceService()

class InferenceRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=1000)
    batch_size: Optional[int] = Field(None, gt=0, le=64)

class HealthResponse(BaseModel):
    status: str
    stats: dict

    class Config:
        protected_namespaces = ()

@app.post("/inference")
async def perform_inference(request: InferenceRequest, background_tasks: BackgroundTasks):
    try:
        if request.batch_size:
            inference_service.batch_size = min(request.batch_size, 64)
        
        # Call the inference method
        results = inference_service.inference(request.texts)
        
        # Log the results for debugging purposes
        logger.info(f"Inference results: {results}")

        # Ensure results are in the correct format
        if not isinstance(results, list):
            raise ValueError("Results should be a list")
        
        # Check that each result is a dictionary and contains expected keys
        for result in results:
            if not isinstance(result, dict):
                raise ValueError(f"Expected dictionary in results, got {type(result)}")
            if "text" not in result or "embedding" not in result:
                raise ValueError(f"Missing keys in result: {result}")

        return {"status": "success", "results": results}
    
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        stats = inference_service.get_health_stats()
        return HealthResponse(status="healthy", stats=stats)
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        return HealthResponse(status="unhealthy", stats={"error": str(e)})

@app.get("/metrics")
async def metrics():
    return {"message": f"Metrics available at http://localhost:{settings.METRICS_PORT}"}

def start_metrics_server():
    try:
        start_http_server(settings.METRICS_PORT)
        logger.info(f"Metrics server started on port {settings.METRICS_PORT}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {str(e)}")
        raise

def run_metrics_server():
    metrics_thread = threading.Thread(target=start_metrics_server, daemon=True)
    metrics_thread.start()

if __name__ == "__main__":
    # Start metrics server in a separate thread
    run_metrics_server()
    
    # Start FastAPI server
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=False  # Disable reload in production
    )
