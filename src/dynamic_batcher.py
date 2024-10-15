import asyncio
from typing import List, Dict, Any
import numpy as np

class DynamicBatcher:
    def __init__(self, max_batch_size: int, max_latency: float):
        self.max_batch_size = max_batch_size
        self.max_latency = max_latency
        self.current_batch: List[Dict[str, Any]] = []
        self.event = asyncio.Event()

    async def add_request(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.current_batch.append(request)
        if len(self.current_batch) >= self.max_batch_size:
            return await self._process_batch()
        elif len(self.current_batch) == 1:
            asyncio.create_task(self._wait_for_batch())
        self.event.set()
        return []

    async def _wait_for_batch(self):
        await asyncio.sleep(self.max_latency)
        if self.current_batch:
            await self._process_batch()

    async def _process_batch(self) -> List[Dict[str, Any]]:
        batch_to_process = self.current_batch
        self.current_batch = []
        self.event.clear()
        return batch_to_process

async def batch_inference(nim_client, model_name: str, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    combined_input = {
        "input_ids": np.concatenate([item["input_ids"] for item in batch]),
        "attention_mask": np.concatenate([item["attention_mask"] for item in batch])
    }
    
    result = await nim_client.inference(model_name, combined_input)
    
    split_results = np.array_split(result["logits"], len(batch))
    return [{"logits": r} for r in split_results]