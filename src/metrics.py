# src/metrics.py
from prometheus_client import Summary, Counter, Gauge, Histogram, CollectorRegistry
import time
import psutil
import GPUtil
import logging
from typing import Optional

class MetricsCollector:
    def __init__(self):
        """Initialize metrics collector with a custom registry."""
        self.logger = logging.getLogger(__name__)
        self.registry = CollectorRegistry()
        
        # Request metrics
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

        # Model metrics
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

        # System metrics
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
        
        # Service health metrics
        self.SERVICE_READY = Gauge(
            'service_ready',
            'Indicates if the service is ready for inference',
            registry=self.registry
        )
        self.MEMORY_USAGE = Gauge(
            'memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )

    def track_request(self) -> float:
        """
        Start tracking a request and return the start time.
        
        Returns:
            float: Start time of the request
        """
        self.REQUEST_COUNT.inc()
        return time.time()
    
    def track_latency(self, start_time: float) -> float:
        """
        Record the latency for a request.
        
        Args:
            start_time (float): Start time of the request
            
        Returns:
            float: Latency in seconds
        """
        latency = time.time() - start_time
        self.REQUEST_LATENCY.observe(latency)
        return latency
    
    def update_gpu_metrics(self, memory_used: Optional[int] = None, utilization: Optional[float] = None):
        """
        Update GPU metrics using either provided values or by querying GPU.
        
        Args:
            memory_used (Optional[int]): GPU memory used in bytes
            utilization (Optional[float]): GPU utilization percentage
        """
        try:
            if memory_used is not None and utilization is not None:
                self.GPU_MEMORY_USED.set(memory_used)
                self.GPU_UTILIZATION.set(utilization)
            else:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Get first GPU
                    self.GPU_MEMORY_USED.set(gpu.memoryUsed * 1024 * 1024)  # Convert MB to bytes
                    self.GPU_UTILIZATION.set(gpu.load * 100)  # Convert to percentage
        except Exception as e:
            self.logger.warning(f"Error updating GPU metrics: {e}")
    
    def track_batch(self, size: int, queue_size: Optional[int] = None):
        """
        Update batch and queue metrics.
        
        Args:
            size (int): Current batch size
            queue_size (Optional[int]): Current queue size
        """
        self.BATCH_SIZE.set(size)
        if queue_size is not None:
            self.QUEUE_SIZE.set(queue_size)
        else:
            self.QUEUE_SIZE.set(size)  # Default to batch size if queue size not provided
    
    def track_inference(self, start_time: float, num_tokens: int = 0) -> float:
        """
        Track inference time and tokens processed.
        
        Args:
            start_time (float): Start time of inference
            num_tokens (int): Number of tokens processed
            
        Returns:
            float: Inference time in seconds
        """
        inference_time = time.time() - start_time
        self.MODEL_INFERENCE_TIME.observe(inference_time)
        if num_tokens > 0:
            self.TOKENS_PROCESSED.inc(num_tokens)
        return inference_time
    
    def update_service_status(self, is_ready: bool):
        """
        Update service readiness status.
        
        Args:
            is_ready (bool): Whether the service is ready for inference
        """
        self.SERVICE_READY.set(1 if is_ready else 0)
        
    def update_memory_usage(self):
        """Update current memory usage metrics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            self.MEMORY_USAGE.set(memory_info.rss)
        except Exception as e:
            self.logger.warning(f"Error updating memory metrics: {e}")
    
    def get_registry(self) -> CollectorRegistry:
        """
        Get the metrics registry.
        
        Returns:
            CollectorRegistry: The collector registry containing all metrics
        """
        return self.registry

# Example usage in FastAPI route:
"""
metrics_collector = MetricsCollector()

@app.post("/inference")
async def inference_endpoint(request: InferenceRequest):
    # Start tracking request
    request_start = metrics_collector.track_request()
    
    # Update GPU metrics
    metrics_collector.update_gpu_metrics()
    
    # Track batch size and queue size
    metrics_collector.track_batch(len(request.texts), current_queue_size)
    
    # Start inference timing
    inference_start = time.time()
    
    # Your inference code here
    result = await model.generate(request.texts)
    
    # Track inference time and tokens
    metrics_collector.track_inference(inference_start, num_tokens=len(result.tokens))
    
    # Track total request latency
    metrics_collector.track_latency(request_start)
    
    return result
"""