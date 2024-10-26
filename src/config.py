# src/confih.py for using the local cpu power
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_NAME: str = "microsoft/MiniLM-L12-H384-uncased"
    BATCH_SIZE: int = 32
    MAX_SEQUENCE_LENGTH: int = 512
    DEVICE: str =  "cpu"  # Hardcoded to CPU
    FALLBACK_TO_CPU: bool = True
    HOST: str = "0.0.0.0"   
    PORT: int = 8000
    METRICS_PORT: int = 8001
    ENABLE_FP16: bool = False
    ENABLE_DYNAMIC_BATCHING: bool = True
    MODEL_PARALLEL: bool = False
    TRITON_URI: str = "localhost:8000"
    
settings = Settings()


# src/config.py for gpu using cuda
# from typing import Optional
# from pydantic_settings import BaseSettings
# import torch  

# class Settings(BaseSettings):
#     MODEL_NAME: str = "microsoft/MiniLM-L12-H384-uncased"
#     BATCH_SIZE: int = 32
#     MAX_SEQUENCE_LENGTH: int = 512
    
#     # Use CUDA if available; fallback to CPU otherwise
#     DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
#     FALLBACK_TO_CPU: bool = True
    
#     HOST: str = "0.0.0.0"
#     PORT: int = 8000
#     METRICS_PORT: int = 8001

#     # Enable FP16 (Half-Precision) if CUDA is available, as it's useful for GPU acceleration
#     ENABLE_FP16: bool = torch.cuda.is_available()
    
#     ENABLE_DYNAMIC_BATCHING: bool = True
#     MODEL_PARALLEL: bool = False
#     TRITON_URI: str = "localhost:8000"

# # Instantiate the settings
# settings = Settings()


#final
# src/confih.py
# from typing import Optional
# from pydantic_settings import BaseSettings
# import torch  # Ensure torch is imported to check for CUDA availability


# class Settings(BaseSettings):
#     MODEL_NAME: str = "microsoft/MiniLM-L12-H384-uncased"
#     BATCH_SIZE: int = 32
#     MAX_SEQUENCE_LENGTH: int = 512
#     DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"  # Dynamically set device
#     FALLBACK_TO_CPU: bool = True
#     HOST: str = "0.0.0.0"   
#     PORT: int = 8000
#     METRICS_PORT: int = 8001
#     ENABLE_FP16: bool = torch.cuda.is_available()  # Enable FP16 if CUDA is available
#     ENABLE_DYNAMIC_BATCHING: bool = True
#     MODEL_PARALLEL: bool = False
#     TRITON_URI: str = "localhost:8000"
    
# settings = Settings()
