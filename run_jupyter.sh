#!/bin/bash

# Build the Docker images if they don't exist
echo "Building Docker images..."

pushd envs/docker/base_image || exit
docker build -t kzkedzierska/sc_foundation_evals:latest .
popd || exit

pushd envs/docker/jupyter || exit
docker build -t sc_foundation_jupyter:latest .
popd || exit

# Run Jupyter in Docker with volume mounting
echo "Starting Jupyter notebook in Docker..."
docker run -it --rm \
    --gpus all \
    -p 8888:8888 \
    -v "$(pwd)":/workspace \
    -v ~/.huggingface:/root/.huggingface \
    sc_foundation_jupyter:latest

echo "Jupyter notebook is running at http://localhost:8888"
echo "Your project files are mounted at /workspace"