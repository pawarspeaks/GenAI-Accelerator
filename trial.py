import torch
from transformers import AutoTokenizer, AutoModel
import os
from pathlib import Path
import onnx
import tensorrt as trt
import numpy as np
import onnxruntime
import pycuda.driver as cuda  # Import PyCUDA for CUDA operations
import pycuda.autoinit  # Initialize CUDA automatically

def inspect_engine(engine_path):
    """Inspect the TensorRT engine properties"""
    logger = trt.Logger(trt.Logger.WARNING)
    
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    print("\nEngine Details:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        
        print(f"{'Input' if is_input else 'Output'} {i}: ")
        print(f"  Name: {name}")
        print(f"  Shape: {shape}")
        print(f"  Datatype: {dtype}")

def infer(engine, input_data):
    """Run inference with TensorRT engine"""
    # Allocate GPU memory for inputs and outputs
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(engine.get_binding_shape(1).volume() * 4)  # Assuming float32 output

    # Copy input to device
    cuda.memcpy_htod(d_input, input_data)

    # Execute the engine
    context = engine.create_execution_context()
    context.execute_v2(bindings=[int(d_input), int(d_output)])

    # Copy output back to host
    output_shape = engine.get_binding_shape(1)  # Get the shape of the output
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)
    
    return output

def convert_model():
    print("Starting model conversion pipeline...")
    
    # Create models directory if it doesn't exist
    Path("models/transformer_model/1").mkdir(parents=True, exist_ok=True)
    
    # Load model
    model_name = "microsoft/MiniLM-L12-H384-uncased"
    print(f"Loading PyTorch model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Save model in Torch format
    print("Saving model in PyTorch format...")
    torch.save(model.state_dict(), "models/transformer_model/1/model.pt")
    tokenizer.save_pretrained("models/transformer_model/1/")
    
    # Convert model to ONNX
    onnx_model_path = "models/transformer_model/1/model.onnx"
    print("Converting model to ONNX format...")
    
    # Use int32 to match config.pbtxt TYPE_INT32
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 64), dtype=torch.int32)
    
    # Export with dynamic axes
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_model_path,
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=17
    )
    print(f"ONNX model saved at {onnx_model_path}")
    
    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification complete.")
    
    # Convert ONNX to TensorRT
    print("Converting ONNX model to TensorRT format...")
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    
    # Create network
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)
    
    # Parse ONNX model
    print("Parsing ONNX model...")
    with open(onnx_model_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(f"TensorRT Parser Error: {parser.get_error(error)}")
            return
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB
    
    # Create optimization profile
    print("Setting up optimization profile...")
    profile = builder.create_optimization_profile()
    
    # Get input tensor name and shape
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    
    # Define ranges for batch and sequence length
    min_batch, min_seq = 1, 16
    opt_batch, opt_seq = 32, 64
    max_batch, max_seq = 64, 128
    
    profile.set_shape(
        input_name,
        min=(min_batch, min_seq),
        opt=(opt_batch, opt_seq),
        max=(max_batch, max_seq)
    )
    
    config.add_optimization_profile(profile)
    
    # Build engine
    print("Building TensorRT engine...")
    engine_bytes = builder.build_serialized_network(network, config)
    
    if engine_bytes is None:
        print("Failed to create engine")
        return
    
    # Save the engine
    print("Saving TensorRT engine...")
    engine_path = "models/transformer_model/1/model.trt"
    try:
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        print(f"TensorRT engine saved successfully at {engine_path}")
        
        # Inspect the saved engine
        inspect_engine(engine_path)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    convert_model()
    
    # Load TensorRT engine and run inference as a test
    engine_path = "models/transformer_model/1/model.trt"
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    # Create some dummy input data for testing
    input_data = np.random.randint(0, 100, (1, 64), dtype=np.int32)
    output = infer(engine, input_data)
    print("Inference output:", output)
