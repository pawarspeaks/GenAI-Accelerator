from nim_client import NIMClient
from typing import Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelDeployer:
    def __init__(self, nim_client: NIMClient):
        self.nim_client = nim_client

    def prepare_and_deploy_model(self, model_name: str, model_path: str) -> Dict[str, Any]:
        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Deploy the model using NIM
        deployment_result = self.nim_client.deploy_model(model_name, model_path)
        return deployment_result

    def inference(self, model_name: str, input_text: str) -> Dict[str, Any]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_data = tokenizer(input_text, return_tensors="np")
        result = self.nim_client.inference(model_name, input_data)
        return result

    def undeploy_model(self, model_name: str) -> Dict[str, Any]:
        return self.nim_client.undeploy_model(model_name)