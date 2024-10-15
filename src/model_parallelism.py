import torch
import torch.nn as nn

class ModelParallelNIMWrapper(nn.Module):
    def __init__(self, nim_client, model_name, num_gpus):
        super().__init__()
        self.nim_client = nim_client
        self.model_name = model_name
        self.num_gpus = num_gpus
        
        # Split the model across GPUs
        self.layer_distribution = self._distribute_layers()

    def _distribute_layers(self):
        # Get model architecture from NIM
        model_info = self.nim_client.get_model_status(self.model_name)
        total_layers = model_info['num_layers']
        
        # Distribute layers evenly across GPUs
        layers_per_gpu = total_layers // self.num_gpus
        return [layers_per_gpu] * self.num_gpus

    def forward(self, input_ids, attention_mask):
        outputs = None
        for gpu_id in range(self.num_gpus):
            device = torch.device(f'cuda:{gpu_id}')
            start_layer = gpu_id * self.layer_distribution[gpu_id]
            end_layer = start_layer + self.layer_distribution[gpu_id]
            
            # Send request to NIM for partial forward pass
            partial_output = self.nim_client.inference(
                self.model_name,
                {
                    'input_ids': input_ids.to(device),
                    'attention_mask': attention_mask.to(device),
                    'start_layer': start_layer,
                    'end_layer': end_layer
                }
            )
            
            if outputs is None:
                outputs = partial_output
            else:
                # Combine outputs from different GPUs
                outputs = self._combine_outputs(outputs, partial_output)
        
        return outputs

    def _combine_outputs(self, outputs1, outputs2):
        # Combine partial outputs from different GPUs
        combined_outputs = {}
        for key in outputs1.keys():
            if isinstance(outputs1[key], torch.Tensor):
                combined_outputs[key] = torch.cat([outputs1[key], outputs2[key]], dim=0)
            elif isinstance(outputs1[key], tuple):
                combined_outputs[key] = tuple(torch.cat([o1, o2], dim=0) for o1, o2 in zip(outputs1[key], outputs2[key]))
            else:
                combined_outputs[key] = outputs1[key]  # Assuming non-tensor outputs are the same across GPUs
        return combined_outputs