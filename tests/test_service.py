# test_service.py
import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_inference():
    try:
        # Test inference endpoint
        inference_response = requests.post(
            "http://localhost:8000/inference",
            json={
                "texts": ["This is a test sentence.", "Another test sentence."]
            }
        )
        logger.info("Inference Response:")
        logger.info(json.dumps(inference_response.json(), indent=2))
        
    except Exception as e:
        logger.error(f"Error testing inference: {str(e)}")

def test_metrics():
    try:
        # Give metrics server time to start
        time.sleep(2)
        
        # Test metrics endpoint
        metrics_response = requests.get("http://localhost:8001/metrics")
        logger.info("Metrics Response:")
        logger.info(metrics_response.text)
        
    except Exception as e:
        logger.error(f"Error testing metrics: {str(e)}")

def test_health():
    try:
        # Test health endpoint
        health_response = requests.get("http://localhost:8000/health")
        logger.info("Health Response:")
        logger.info(json.dumps(health_response.json(), indent=2))
        
    except Exception as e:
        logger.error(f"Error testing health: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting tests...")
    test_inference()
    test_metrics()
    test_health()