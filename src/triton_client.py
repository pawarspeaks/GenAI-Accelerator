import tritonclient.http as triton_http
from tritonclient.utils import InferenceServerException

class TritonClient:
    def __init__(self, url="localhost:8000"):
        self.client = triton_http.InferenceServerClient(url)
        
    def infer(self, input_data):
        inputs = []
        outputs = []
        # Configure inputs/outputs based on model
        response = self.client.infer("model_name", inputs)
        return response