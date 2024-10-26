# src/triton_client.py
import tritonclient.http as triton_http
from tritonclient.utils import InferenceServerException
from typing import List  # Import List from the typing module

class TritonClient:
    def __init__(self, url="localhost:8000"):
        self.client = triton_http.InferenceServerClient(url)
        
    def infer(self, model_name: str, inputs: List[triton_http.InferInput]):
        """Perform inference on a specified model with the provided inputs."""
        try:
            response = self.client.infer(model_name, inputs)
            return response
        except InferenceServerException as e:
            print(f"Error during inference: {e}")
            raise



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