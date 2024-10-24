import tensorrt as trt
import pycuda.driver as cuda

class TensorRTClient:
    def __init__(self):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
    def load_engine(self, path):
        with open(path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())