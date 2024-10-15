import streamlit as st
import requests
import json

BASE_URL = "http://localhost:8080"

st.title("GenAI Accelerator Dashboard")

# Function to handle both JSON and non-JSON responses
def handle_response(response):
    st.text(f"Status Code: {response.status_code}")
    st.text(f"Raw Response Content: {response.text}")
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
if st.button("Deploy Model"):
    response = requests.get(f"{BASE_URL}/deploy", params={"model_name": model_name, "model_path": model_path})
    handle_response(response)

# Inference
st.header("Inference")
inference_model = st.text_input("Model Name for Inference")
input_text = st.text_area("Input Text")
if st.button("Run Inference"):
    # Use GET request instead of POST, with query parameters
    response = requests.get(f"{BASE_URL}/inference", params={"model_name": inference_model, "input_text": input_text})
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
    handle_response(response)

# Benchmarking
st.header("Benchmarking")
benchmark_model = st.text_input("Model Name for Benchmarking")
num_iterations = st.number_input("Number of Iterations", min_value=1, value=100)
concurrency = st.number_input("Concurrency", min_value=1, value=10)
if st.button("Run Benchmark"):
    st.write("Benchmarking started. This may take a while...")
    # Placeholder for benchmark results until the real results are available
    st.json({
        "average_latency": 0.1,
        "p50_latency": 0.09,
        "p95_latency": 0.15,
        "p99_latency": 0.2,
        "average_throughput": 100,
        "max_throughput": 120,
    })
