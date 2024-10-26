from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Summary, Counter, Gauge, Histogram,
    generate_latest, CollectorRegistry
)
import time
import logging
from .nim_service import NIMInferenceService
from .config import settings

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

class Metrics:
    def __init__(self):
        # Create a new registry
        self.registry = CollectorRegistry()
        
        # Initialize metrics with the custom registry
        self.REQUEST_LATENCY = Histogram(
            'request_latency_seconds',
            'Request latency in seconds',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0),
            registry=self.registry
        )
        
        self.REQUEST_COUNT = Counter(
            'request_count',
            'Total request count',
            registry=self.registry
        )
        
        self.BATCH_SIZE = Gauge(
            'batch_size_current',
            'Current batch size',
            registry=self.registry
        )
        
        self.GPU_MEMORY_USED = Gauge(
            'gpu_memory_used_bytes',
            'GPU memory used in bytes',
            registry=self.registry
        )
        
        self.MODEL_INFERENCE_TIME = Summary(
            'model_inference_seconds',
            'Time spent on model inference',
            registry=self.registry
        )
        
        self.TOKENS_PROCESSED = Counter(
            'tokens_processed_total',
            'Total number of tokens processed',
            registry=self.registry
        )
        
        self.QUEUE_SIZE = Gauge(
            'inference_queue_size',
            'Number of requests in queue',
            registry=self.registry
        )
        
        self.GPU_UTILIZATION = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            registry=self.registry
        )

# Create a global metrics instance
metrics = Metrics()

class MetricsCollector:
    @staticmethod
    def track_request():
        metrics.REQUEST_COUNT.inc()
        return time.time()
    
    @staticmethod
    def track_latency(start_time):
        metrics.REQUEST_LATENCY.observe(time.time() - start_time)
    
    @staticmethod
    def update_gpu_metrics(memory_used, utilization):
        metrics.GPU_MEMORY_USED.set(memory_used)
        metrics.GPU_UTILIZATION.set(utilization)
    
    @staticmethod
    def track_batch(size):
        metrics.BATCH_SIZE.set(size)

# Rest of your code remains the same, but use 'metrics' instead of 'Metrics'
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
    start_time = MetricsCollector.track_request()
    try:
        if request.batch_size:
            inference_service.batch_size = min(request.batch_size, 64)
            MetricsCollector.track_batch(inference_service.batch_size)
        
        # Start model inference timing
        with metrics.MODEL_INFERENCE_TIME.time():
            results = inference_service.inference(request.texts)
        
        # Update tokens processed
        metrics.TOKENS_PROCESSED.inc(len(request.texts))
        
        # Rest of your inference logic...
        return {"status": "success", "results": results}
    
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        MetricsCollector.track_latency(start_time)

@app.get("/health")
async def health_check():
    start_time = MetricsCollector.track_request()
    try:
        stats = inference_service.get_health_stats()
        if hasattr(inference_service, 'get_gpu_stats'):
            gpu_stats = inference_service.get_gpu_stats()
            MetricsCollector.update_gpu_metrics(
                gpu_stats.get('memory_used', 0),
                gpu_stats.get('utilization', 0)
            )
        return HealthResponse(status="healthy", stats=stats)
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        return HealthResponse(status="unhealthy", stats={"error": str(e)})
    finally:
        MetricsCollector.track_latency(start_time)

@app.get("/metrics")
async def metrics_endpoint():
    """
    Endpoint that serves Prometheus metrics
    """
    return PlainTextResponse(
        generate_latest(metrics.registry),
        media_type=CONTENT_TYPE_LATEST
    )

if __name__ == "__main__":
    # Start FastAPI server
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=False  # Disable reload in production
    )