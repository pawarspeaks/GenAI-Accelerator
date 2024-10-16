FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Step 1: Update and install Python3, pip, and cmake in one layer
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Set the working directory
WORKDIR /app

# Step 3: Create logs directory and set permissions
RUN mkdir -p logs && chmod 777 logs

# Step 4: Copy the requirements.txt file first to leverage caching
COPY requirements.txt ./

# Step 5: Install Python packages from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of the application code
COPY . .

# Step 7: Specify the command to run your application
CMD ["python3", "src/main.py"]
