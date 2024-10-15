import asyncio
import time
import aiohttp
from typing import List, Dict
import numpy as np
from tqdm import tqdm

class Benchmarker:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def run_inference(self, model_name: str, input_text: str) -> Dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/inference", json={"model_name": model_name, "input_text": input_text}) as response:
                return await response.json()

    async def run_benchmark(self, model_name: str, input_texts: List[str], num_iterations: int = 100, concurrency: int = 10):
        latencies = []
        throughputs = []

        async def worker(input_text: str):
            start_time = time.time()
            await self.run_inference(model_name, input_text)
            end_time = time.time()
            return end_time - start_time

        for _ in tqdm(range(num_iterations)):
            start_time = time.time()
            tasks = [worker(text) for text in input_texts[:concurrency]]
            batch_latencies = await asyncio.gather(*tasks)
            end_time = time.time()

            latencies.extend(batch_latencies)
            throughput = len(batch_latencies) / (end_time - start_time)
            throughputs.append(throughput)

        return {
            "average_latency": np.mean(latencies),
            "p50_latency": np.percentile(latencies, 50),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "average_throughput": np.mean(throughputs),
            "max_throughput": np.max(throughputs),
        }

async def main():
    benchmarker = Benchmarker("http://localhost:8080")
    model_name = "gpt2"
    input_texts = [
        "Hello, how are you?",
        "What's the weather like today?",
        "Can you tell me a joke?",
        "What's the capital of France?",
        "How do I make a pizza?",
    ]
    results = await benchmarker.run_benchmark(model_name, input_texts)
    print("Benchmark Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())