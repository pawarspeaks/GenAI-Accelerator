FROM nvcr.io/nvidia/tritonserver:23.10-py3

# Install TensorRT
RUN apt-get update && apt-get install -y python3-libnvinfer

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .
COPY models/ /models

EXPOSE 8000 8001 8002

CMD ["tritonserver", "--model-repository=/models"]