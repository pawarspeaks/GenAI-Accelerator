from setuptools import setup, find_packages

setup(
    name="genai-accelerator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.68.0",
        "uvicorn==0.15.0",
        "pydantic==1.8.2",
        "requests==2.26.0",
        "torch==1.9.0",
        "transformers==4.10.0",
        "onnx==1.10.1",
        "onnxruntime==1.8.1",
        "kubernetes==18.20.0",
        "prometheus-client==0.11.0",
        "psutil==5.8.0",
        "GPUtil==1.4.0",
        "numpy==1.21.2",
        "aiohttp==3.7.4",
        "tqdm==4.62.2",
        "streamlit==0.88.0",
        "plotly==5.3.1",
        "pandas==1.3.3",
    ],
    entry_points={
        "console_scripts": [
            "genai-accelerator=src.main:main",
        ],
    },
)