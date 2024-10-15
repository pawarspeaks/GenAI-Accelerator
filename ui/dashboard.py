import streamlit as st
import requests
import json
import plotly.graph_objs as go
import pandas as pd

BASE_URL = "http://localhost:8080"

st.title("GenAI Accelerator Dashboard")

def handle_response(response):
    st.text(f"Status Code: {response.status_code}")
    if response.headers.get('Content-Type') == 'application/json':
        try:
            st.json(response.json())
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON: {str(e)}")
    else:
        st.text("Non-JSON response content:")
        st.text(response.text)

# Model Deployment
st.header("Model Deployment")
model_name = st.text_input("Model Name")
model_path = st.text_input("Model Path")
num_gpus = st.number_input("Number of GPUs", min_value=1, value=1)
if st.button("Deploy Model"):
    response = requests.post(f"{BASE_URL}/deploy", json={"model_name": model_name, "model_path": model_path, "num_gpus": num_gpus})
    handle_response(response)

# Inference
st.header("Inference")
inference_model = st.text_input("Model Name for Inference")
input_text = st.text_area("Input Text")
if st.button("Run Inference"):
    response = requests.post(f"{BASE_URL}/inference", json={"model_name": inference_model, "input_text": input_text})
    handle_response(response)

# Model Undeployment
st.header("Model Undeployment")
undeploy_model = st.text_input("Model Name to Undeploy")
if st.button("Undeploy Model"):
    response = requests.post(f"{BASE_URL}/undeploy/{undeploy_model}")
    handle_response(response)

# Metrics
st.header("Metrics")
if st.button("Refresh Metrics"):
    response = requests.get(f"{BASE_URL}/metrics")
    metrics = response.json()
    
    # Create metrics visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Inference Requests", metrics.get("inference_requests_total", 0))
        st.metric("Average Latency", f"{metrics.get('model_inference_latency_seconds', 0):.2f}s")
    with col2:
        st.metric("Current Batch Size", metrics.get("current_batch_size", 0))
        st.metric("GPU Utilization", f"{metrics.get('gpu_utilization_percent', 0):.1f}%")

    # Create a line chart for memory usage over time
    memory_data = metrics.get("memory_usage_bytes", [])
    if memory_data:
        df = pd.DataFrame(memory_data, columns=["timestamp", "memory_usage"])
        fig = go.Figure(data=go.Scatter(x=df["timestamp"], y=df["memory_usage"]))
        fig.update_layout(title="Memory Usage Over Time", xaxis_title="Time", yaxis_title="Memory Usage (bytes)")
        st.plotly_chart(fig)

# Scaling
st.header("Scaling")
scale_name = st.text_input("Deployment Name")
scale_namespace = st.text_input("Namespace")
scale_replicas = st.number_input("Number of Replicas", min_value=1, value=1)
if st.button("Scale Deployment"):
    response = requests.post(f"{BASE_URL}/scale", params={"name": scale_name, "namespace": scale_namespace, "replicas": scale_replicas})
    handle_response(response)

# Update HPA
st.header("Update HPA")
hpa_name = st.text_input("HPA Name")
hpa_namespace = st.text_input("HPA Namespace")
min_replicas = st