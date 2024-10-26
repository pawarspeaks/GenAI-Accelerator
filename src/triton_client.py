# src/triton_client.py
import tritonclient.http as triton_http
from tritonclient.utils import InferenceServerException
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict

class TritonClient:
    def __init__(self, model_name: str, device: str = "cpu", triton_url="localhost:8000"):
        """
        Initialize TritonClient for both CPU and GPU inference.
        
        Args:
            model_name (str): The model name for loading.
            device (str): "cpu" or "cuda" based on desired inference device.
            triton_url (str): URL of the Triton inference server.
        """
        self.device = device
        self.model_name = model_name
        
        # Initialize Triton client for GPU
        if device == "cuda":
            self.client = triton_http.InferenceServerClient(url=triton_url)
        else:
            # For CPU, load model locally
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()

    def infer(self, texts: List[str]) -> List[Dict]:
        """
        Perform inference on the model with given texts.
        
        Args:
            texts (List[str]): List of input texts to process.
        
        Returns:
            List[Dict]: List of inference results with embeddings.
        """
        if self.device == "cuda":
            # GPU Inference via Triton
            inputs = self.preprocess_for_triton(texts)
            try:
                response = self.client.infer(self.model_name, inputs)
                embeddings = response.as_numpy("output")  # Ensure this matches your model's output tensor name
                return [{"text": text, "embedding": embedding.tolist()} for text, embedding in zip(texts, embeddings)]
            except InferenceServerException as e:
                print(f"Error during inference: {e}")
                raise

        else:
            # CPU Inference via PyTorch
            encodings = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt", max_length=512
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model(**encodings)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Taking mean of hidden states
            
            return [{"text": text, "embedding": embedding.tolist()} for text, embedding in zip(texts, embeddings)]

    def preprocess_for_triton(self, texts: List[str]) -> List[triton_http.InferInput]:
        """
        Preprocess texts into Triton InferInput format for GPU inference.
        
        Args:
            texts (List[str]): List of input texts to process.
        
        Returns:
            List[triton_http.InferInput]: List of Triton InferInput objects.
        """
        # Tokenize using the tokenizer
        encodings = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        )

        # Convert each input to Triton InferInput
        triton_inputs = []
        for name, tensor in encodings.items():
            # Adjust data type as needed, make sure it matches your Triton model's input format
            infer_input = triton_http.InferInput(name, tensor.shape, "INT32")  # Assuming input is INT32, adjust if needed
            infer_input.set_data_from_numpy(tensor.numpy())
            triton_inputs.append(infer_input)
        
        return triton_inputs

# import tritonclient.http as triton_http
# from tritonclient.utils import InferenceServerException
# import numpy as np
# import logging
# import time

# class TritonClient:
#     def __init__(self, url="localhost:8000", timeout=30.0, retry_count=3):
#         self.url = url
#         self.timeout = timeout
#         self.retry_count = retry_count
#         self.logger = logging.getLogger(__name__)
#         self._init_client()
        
#     def _init_client(self):
#         """Initialize the Triton client with connection checking."""
#         try:
#             self.client = triton_http.InferenceServerClient(
#                 url=self.url,
#                 verbose=True,
#                 connection_timeout=self.timeout,
#                 network_timeout=self.timeout
#             )
#             # Check server readiness
#             if not self.is_server_ready():
#                 raise ConnectionError("Triton server is not ready")
#         except Exception as e:
#             self.logger.error(f"Failed to initialize Triton client: {str(e)}")
#             raise
            
#     def is_server_ready(self):
#         """Check if the Triton server is ready."""
#         try:
#             return self.client.is_server_ready()
#         except Exception as e:
#             self.logger.error(f"Server readiness check failed: {str(e)}")
#             return False
            
#     def is_model_ready(self, model_name):
#         """Check if the specified model is ready."""
#         try:
#             return self.client.is_model_ready(model_name)
#         except Exception as e:
#             self.logger.error(f"Model readiness check failed for {model_name}: {str(e)}")
#             return False

#     def infer(self, model_name, inputs, retry_count=None):
#         """Perform inference with retry logic.

#         Args:
#             model_name (str): The name of the model to use for inference.
#             inputs (list): List of tuples containing input name and numpy data.
#             retry_count (int): Number of retries for failed requests.

#         Returns:
#             response: The response from the Triton Inference Server.
#         """
#         if retry_count is None:
#             retry_count = self.retry_count

#         # Check server and model readiness
#         if not self.is_server_ready():
#             raise ConnectionError("Triton server is not ready")
#         if not self.is_model_ready(model_name):
#             raise ValueError(f"Model {model_name} is not ready")

#         infer_inputs = []
#         last_exception = None

#         # Validate and prepare inputs
#         try:
#             for item in inputs:
#                 if isinstance(item, tuple) and len(item) == 2:
#                     input_name, input_data = item
                    
#                     # Ensure data is float32
#                     if input_data.dtype != np.float32:
#                         self.logger.warning(f"Converting input {input_name} from {input_data.dtype} to float32")
#                         input_data = input_data.astype(np.float32)
                    
#                     self.logger.info(f"Creating inference input: {input_name}, shape: {input_data.shape}, dtype: {input_data.dtype}")
#                     infer_input = triton_http.InferInput(input_name, input_data.shape, "FP32")
#                     infer_input.set_data_from_numpy(input_data)
#                     infer_inputs.append(infer_input)
#                 else:
#                     raise ValueError(f"Expected tuple of (input_name, input_data), got: {item}")

#             output = triton_http.InferRequestedOutput("output")
#         except Exception as e:
#             self.logger.error(f"Error preparing inputs: {str(e)}")
#             raise

#         # Retry logic for inference
#         for attempt in range(retry_count):
#             try:
#                 self.logger.info(f"Sending inference request to model: {model_name} (attempt {attempt + 1}/{retry_count})")
#                 response = self.client.infer(
#                     model_name,
#                     inputs=infer_inputs,
#                     outputs=[output],
#                     client_timeout=self.timeout
#                 )
#                 return response
#             except InferenceServerException as e:
#                 last_exception = e
#                 self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
#                 if attempt < retry_count - 1:
#                     time.sleep(min(2 ** attempt, 10))  # Exponential backoff
#                 continue
#             except Exception as e:
#                 self.logger.error(f"Unexpected error during inference: {str(e)}")
#                 raise

#         # If all retries failed
#         self.logger.error(f"All {retry_count} inference attempts failed")
#         raise last_exception or Exception("Inference failed after all retries")