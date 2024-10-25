# src/nim_service.py
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Optional
import logging
from .metrics import MetricsCollector
from .model_optimizer import ModelOptimizer
from .config import settings
from .triton_client import TritonClient
from .tensorrt_client import TensorRTClient

class NIMInferenceService:
    def __init__(self):
        """Initialize the NIM Inference Service with model setup and optimization."""
        self.setup_logging()
        self.setup_device()
        self.setup_model()
        self.setup_optimizer()
        self.metrics = MetricsCollector()
        
    def setup_logging(self):
        """Configure logging for the service."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_device(self):
        """Set up the compute device (CPU/GPU) based on availability and settings."""
        self.device = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def setup_model(self):
        """Set up the model based on device type with fallback to CPU if needed."""
        try:
            if self.device.type == "cpu":
                self.logger.info("Running in CPU mode")
                self._setup_cpu_model()
            else:
                self.logger.info("Running in GPU mode")
                self._setup_gpu_model()
        except Exception as e:
            if settings.FALLBACK_TO_CPU:
                self.logger.warning(f"Failed to setup GPU model: {e}. Falling back to CPU")
                self.device = torch.device("cpu")
                self._setup_cpu_model()
            else:
                raise

    def _setup_cpu_model(self):
        """Set up model for CPU inference."""
        self.logger.info("Loading model for CPU inference")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        self.model = AutoModel.from_pretrained(settings.MODEL_NAME)
        
        # Set padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.batch_size = settings.BATCH_SIZE
        self.logger.info(f"CPU setup complete. Using batch size: {self.batch_size}")

    def _setup_gpu_model(self):
        """Set up model for GPU inference using Triton and TensorRT."""
        self.logger.info("Setting up GPU model with Triton and TensorRT")
        
        # Initialize NIM clients
        self.triton_client = TritonClient(settings.TRITON_URI)
        self.tensorrt_client = TensorRTClient()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Calculate optimal batch size
        available_memory = torch.cuda.get_device_properties(0).total_memory
        self.batch_size = ModelOptimizer.get_optimal_batch_size(
            sum(p.numel() for p in self.model.parameters()),
            available_memory
        )
        
        self.logger.info(f"GPU setup complete. Using batch size: {self.batch_size}")
        
    def setup_optimizer(self):
        """Set up model optimization based on device type."""
        if hasattr(self, 'model'):  # Only for CPU mode
            optimizer = ModelOptimizer(
                self.model,
                enable_fp16=settings.ENABLE_FP16 and self.device.type == "cuda"
            )
            self.model = optimizer.optimize()
        
    def preprocess_batch(self, texts: List[str]) -> Dict:
        """
        Preprocess a batch of texts for inference.
        
        Args:
            texts (List[str]): List of input texts to process
            
        Returns:
            Dict: Preprocessed inputs ready for model inference
        """
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=settings.MAX_SEQUENCE_LENGTH,
            return_tensors="pt"
        )
        
        if self.device.type == "cuda":
            return {k: v.to(self.device) for k, v in encodings.items()}
        return encodings
        
    @torch.inference_mode()
    def inference(self, texts: List[str]) -> List[Dict]:
        """
        Perform inference on a list of input texts.
        
        Args:
            texts (List[str]): List of input texts to process
            
        Returns:
            List[Dict]: List of results containing text and embeddings
        """
        start_time = self.metrics.track_request()
        results = []
        
        try:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                self.metrics.track_batch(len(batch_texts))
                
                # Preprocess batch
                inputs = self.preprocess_batch(batch_texts)
                
                # Run inference based on device
                if self.device.type == "cpu":
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state.numpy()
                else:
                    # Use Triton/TensorRT for GPU inference
                    outputs = self.triton_client.infer(
                        model_name="transformer_model",
                        inputs=inputs
                    )
                    embeddings = outputs.as_numpy("output")
                
                # Process results
                for j, text in enumerate(batch_texts):
                    results.append({
                        "text": text,
                        "embedding": embeddings[j].mean(axis=0).tolist()
                    })
                
                # Update metrics
                if self.device.type == "cuda":
                    memory_used = torch.cuda.memory_allocated()
                    utilization = torch.cuda.utilization()
                    self.metrics.update_gpu_metrics(memory_used, utilization)
                    
            return results
            
        finally:
            self.metrics.track_latency(start_time)
            
    def get_health_stats(self) -> Dict:
        """Get health statistics about the service."""
        stats = {
            "model_name": settings.MODEL_NAME,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "fp16_enabled": settings.ENABLE_FP16 and self.device.type == "cuda",
            "max_sequence_length": settings.MAX_SEQUENCE_LENGTH
        }
        
        if self.device.type == "cuda":
            stats.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_reserved": torch.cuda.memory_reserved(),
                "gpu_name": torch.cuda.get_device_name()
            })
            
        return stats