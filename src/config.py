from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_NAME: str = "microsoft/MiniLM-L12-H384-uncased"
    BATCH_SIZE: int = 32
    MAX_SEQUENCE_LENGTH: int = 512
    DEVICE: str =  "cpu"
    FALLBACK_TO_CPU: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    METRICS_PORT: int = 8001
    ENABLE_FP16: bool = False
    ENABLE_DYNAMIC_BATCHING: bool = True
    MODEL_PARALLEL: bool = False
    TRITON_URI: str = "localhost:8000"
    
settings = Settings()


# from typing import Optional
# from pydantic_settings import BaseSettings


# class Settings(BaseSettings):
#     MODEL_NAME: str = "microsoft/MiniLM-L12-H384-uncased"
#     BATCH_SIZE: int = 32
#     MAX_SEQUENCE_LENGTH: int = 512
#     DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
#     FALLBACK_TO_CPU: bool = True
#     HOST: str = "0.0.0.0"
#     PORT: int = 8000
#     METRICS_PORT: int = 8001
#     ENABLE_FP16: bool = torch.cuda.is_available()
#     ENABLE_DYNAMIC_BATCHING: bool = True
#     MODEL_PARALLEL: bool = False
#     TRITON_URI: str = "localhost:8000"
    
# settings = Settings()