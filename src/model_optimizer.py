# src/model_optimizer.py
import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ModelOptimizer:
    def __init__(self, model: nn.Module, enable_fp16: bool = True):
        self.model = model
        self.enable_fp16 = enable_fp16
        
    def optimize(self):
        """Optimize model for inference"""
        self.model.eval()
        
        if torch.cuda.is_available() and self.enable_fp16:
            logger.info("Enabling FP16 inference")
            self.model.half()
            
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        return self.model
        
    @staticmethod
    def get_optimal_batch_size(model_params: int, available_memory: int) -> int:
        """Calculate optimal batch size based on model size and available memory"""
        # Rough heuristic: assume each parameter needs 4 bytes
        memory_per_sample = model_params * 4
        optimal_batch_size = max(1, available_memory // (memory_per_sample * 2))
        return min(optimal_batch_size, 64)  # Cap at 64 to avoid memory issues