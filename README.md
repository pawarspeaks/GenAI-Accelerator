# GenAI Accelerator

This project leverages NVIDIA Inference Microservices (NIM) to scale and optimize Generative AI applications. It provides a high-performance solution for deploying and managing large language models and other generative AI systems.

## Features

- Integration with NIM microservices for model deployment and scaling
- Dynamic batching and model parallelism for efficient inference
- Real-time monitoring and auto-scaling based on inference demand
- Performance metrics for measuring latency, throughput, and resource utilization
- User-friendly dashboard for managing and monitoring Gen AI models

## Prerequisites

- Docker
- Kubernetes
- Helm
- NVIDIA GPU with CUDA support
- Python 3.8+

## Installation

1. Clone the repository:

git clone https://github.com/pawarspeaks/GenAI-Accelerator.git
cd GenAI-Accelerator

2. Install the required Python packages:

pip install -r requirements.txt

3. Set up the configuration:
Edit `config/config.yaml` to match your environment and requirements.

## Usage

1. Start the GenAI Accelerator service:

python src/main.py

2. Access the dashboard:

streamlit run ui/dashboard.py

3. Use the API endpoints for model deployment, inference, and management.

## Running Tests

To run the unit tests:

python -m unittest discover tests

## Benchmarking

To run performance benchmarks:

python benchmarks/benchmark.py

## Deployment

For production deployment, use the provided Kubernetes and Helm charts:

kubectl apply -f kubernetes/ -n genai-accelerator

or

helm install genai-accelerator helm/


