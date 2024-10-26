# src/config.py
from typing import Optional
from pydantic_settings import BaseSettings
import torch  # Ensure torch is imported to check for CUDA availability

class Settings(BaseSettings):
    # Model settings
    MODEL_NAME: str = "microsoft/MiniLM-L12-H384-uncased"
    BATCH_SIZE: int = 32
    MAX_SEQUENCE_LENGTH: int = 512
    
    # Device settings: Select "cuda" if available, otherwise "cpu"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    FALLBACK_TO_CPU: bool = True
    
    # CPU-specific default batch size
    DEFAULT_CPU_BATCH_SIZE: int = 8

    # Host and port settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    METRICS_PORT: int = 8001

    # GPU-specific settings
    ENABLE_FP16: bool = torch.cuda.is_available()  # Enable FP16 only if CUDA is available
    ENABLE_DYNAMIC_BATCHING: bool = True
    MODEL_PARALLEL: bool = False
    
    # Triton Inference Server URI
    TRITON_URI: str = "localhost:8000"

# Instantiate the settings
settings = Settings()
