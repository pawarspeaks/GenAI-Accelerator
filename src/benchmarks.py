import time
import numpy as np

class BenchmarkRunner:
    def __init__(self, service):
        self.service = service
        self.results = []
        
    def run_benchmark(self, texts, batch_sizes=[1, 4, 8, 16, 32]):
        results = []
        for batch_size in batch_sizes:
            latencies = []
            for _ in range(10):  # 10 runs per batch size
                start = time.time()
                self.service.inference(texts[:batch_size])
                latencies.append(time.time() - start)
            
            results.append({
                "batch_size": batch_size,
                "avg_latency": np.mean(latencies),
                "throughput": batch_size / np.mean(latencies)
            })
        return results
