from prometheus_client import Summary, Counter, Gauge, Histogram
import time

# Request metrics
REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Request latency in seconds',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
)
REQUEST_COUNT = Counter('request_count', 'Total request count')

# Model metrics
BATCH_SIZE = Gauge('batch_size_current', 'Current batch size')
GPU_MEMORY_USED = Gauge('gpu_memory_used_bytes', 'GPU memory used in bytes')
MODEL_INFERENCE_TIME = Summary('model_inference_seconds', 'Time spent on model inference')
TOKENS_PROCESSED = Counter('tokens_processed_total', 'Total number of tokens processed')

# System metrics
QUEUE_SIZE = Gauge('inference_queue_size', 'Number of requests in queue')
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization percentage')

class MetricsCollector:
    @staticmethod
    def track_request():
        REQUEST_COUNT.inc()
        return time.time()
    
    @staticmethod
    def track_latency(start_time):
        REQUEST_LATENCY.observe(time.time() - start_time)
    
    @staticmethod
    def update_gpu_metrics(memory_used, utilization):
        GPU_MEMORY_USED.set(memory_used)
        GPU_UTILIZATION.set(utilization)
    
    @staticmethod
    def track_batch(size):
        BATCH_SIZE.set(size)