import torch
from transformers import AutoTokenizer, AutoModel
import os
from pathlib import Path

def convert_model():
    print("Starting model conversion...")
    
    # Create models directory if it doesn't exist
    Path("models/transformer_model/1").mkdir(parents=True, exist_ok=True)
    
    # Load model
    model_name = "microsoft/MiniLM-L12-H384-uncased"
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Save model in Torch format first
    print("Saving model in Torch format...")
    torch.save(model.state_dict(), "models/transformer_model/1/model.pt")
    tokenizer.save_pretrained("models/transformer_model/1/")
    
    # Create config file
    config_content = """name: "transformer_model"
platform: "pytorch_libtorch"
max_batch_size: 64
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]
output [
  {
    name: "last_hidden_state"
    data_type: TYPE_FP32
    dims: [ -1, 384 ]
  }
]
"""
    
    with open("models/transformer_model/config.pbtxt", "w") as f:
        f.write(config_content)
    
    print("Model conversion completed successfully!")

if __name__ == "__main__":
    convert_model()