#!/bin/bash

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "ERROR: Docker is not running!"
    echo ""
    echo "Please ensure Docker is started and accessible, then try again."
    echo "For installation and setup instructions, see: https://docs.docker.com/get-docker/"
    echo "For GPU support, also ensure NVIDIA Container Toolkit is installed: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
    exit 1
fi

# Try to pull the Docker image first, build as fallback
echo "Attempting to pull pre-built Docker image..."
if docker pull kzkedzierska/sc_foundation_evals:latest_notebook &> /dev/null; then
    echo "Using pre-built Docker image"
    IMAGE_NAME="kzkedzierska/sc_foundation_evals:latest_notebook"
else
    echo "Pre-built image not available, building locally..."
    
    # Build base image
    pushd envs/docker/base_image || exit 1
    if ! docker build -t kzkedzierska/sc_foundation_evals:latest .; then
        echo "ERROR: Failed to build base Docker image"
        exit 1
    fi
    popd || exit 1
    
    # Build jupyter image
    pushd envs/docker/jupyter || exit 1
    if ! docker build -t sc_foundation_jupyter:latest .; then
        echo "ERROR: Failed to build Jupyter Docker image"
        exit 1
    fi
    popd || exit 1
    
    echo "Local build completed successfully!"
    IMAGE_NAME="sc_foundation_jupyter:latest"
fi

# Run Jupyter in Docker with volume mounting
echo "Starting Jupyter notebook in Docker..."
docker run -it --rm \
    --gpus all \
    -p 8888:8888 \
    -v "$(pwd)":/workspace \
    -v ~/.huggingface:/root/.huggingface \
    "$IMAGE_NAME"