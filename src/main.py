import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from nim_client import NIMClient
from model_deployer import ModelDeployer
from dynamic_batcher import DynamicBatcher, batch_inference
from model_parallelism import ModelParallelNIMWrapper
from monitoring import Monitoring
from auto_scaler import AutoScaler
import yaml
import asyncio
import time
import os

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

nim_client = NIMClient(config["nim_base_url"])
model_deployer = ModelDeployer(nim_client)
dynamic_batcher = DynamicBatcher(config["max_batch_size"], config["max_latency"])
monitoring = Monitoring()
auto_scaler = AutoScaler()

class DeployRequest(BaseModel):
    model_name: str
    model_path: str
    num_gpus: int = 1

class InferenceRequest(BaseModel):
    model_name: str
    input_text: str

async def update_metrics_periodically():
    while True:
        monitoring.collect_metrics()
        await asyncio.sleep(60)  # Update metrics every 60 seconds

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    asyncio.create_task(update_metrics_periodically())
    yield
    # Shutdown
    # Add any cleanup code here if needed

app = FastAPI(lifespan=lifespan)

@app.post("/deploy")
async def deploy_model(request: DeployRequest):
    try:
        result = await model_deployer.prepare_and_deploy_model(request.model_name, request.model_path)
        if request.num_gpus > 1:
            # Initialize model parallelism
            ModelParallelNIMWrapper(nim_client, request.model_name, request.num_gpus)
        return {"message": "Model deployed successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference")
async def inference(request: InferenceRequest):
    try:
        monitoring.record_inference_request()
        
        @monitoring.measure_request_time
        async def process_inference():
            tokenized_input = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}  # Mocked tokenization
            batch = await dynamic_batcher.add_request(tokenized_input)
            if batch:
                monitoring.update_batch_size(len(batch))
                start_time = time.time()
                results = await batch_inference(nim_client, request.model_name, batch)
                latency = time.time() - start_time
                monitoring.record_model_latency(latency)
                return results[0]  # Return the result for this specific request
            return None

        result = await process_inference()
        if result is None:
            return {"message": "Request added to batch, please try again later"}
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/undeploy/{model_name}")
async def undeploy_model(model_name: str):
    try:
        result = await model_deployer.undeploy_model(model_name)
        return {"message": "Model undeployed successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    monitoring.collect_metrics()
    return {"message": "Metrics updated"}

@app.post("/scale")
async def scale_deployment(name: str, namespace: str, replicas: int):
    try:
        auto_scaler.scale_deployment(name, namespace, replicas)
        return {"message": f"Scaled deployment {name} to {replicas} replicas"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_hpa")
async def update_hpa(name: str, namespace: str, min_replicas: int, max_replicas: int, target_cpu_utilization: int):
    try:
        auto_scaler.update_hpa(name, namespace, min_replicas, max_replicas, target_cpu_utilization)
        return {"message": f"Updated HPA for deployment {name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)