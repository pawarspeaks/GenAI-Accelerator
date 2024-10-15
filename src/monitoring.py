from prometheus_client import start_http_server, Summary, Counter, Gauge
import time
import psutil
import GPUtil

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
INFERENCE_REQUESTS = Counter('inference_requests_total', 'Total number of inference requests')
MODEL_LATENCY = Summary('model_inference_latency_seconds', 'Latency of model inference')
BATCH_SIZE = Gauge('current_batch_size', 'Current size of the inference batch')
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')

class Monitoring:
    def __init__(self, port=8000):
        start_http_server(port)

    @REQUEST_TIME.time()
    def measure_request_time(self, func):
        return func()

    def record_inference_request(self):
        INFERENCE_REQUESTS.inc()

    def record_model_latency(self, latency):
        MODEL_LATENCY.observe(latency)

    def update_batch_size(self, size):
        BATCH_SIZE.set(size)

    def update_gpu_utilization(self):
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            GPU_UTILIZATION.labels(gpu=i).set(gpu.load * 100)

    def update_memory_usage(self):
        MEMORY_USAGE.set(psutil.virtual_memory().used)

    def collect_metrics(self):
        self.update_gpu_utilization()
        self.update_memory_usage()