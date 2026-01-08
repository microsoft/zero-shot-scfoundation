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

# Check GPU compatibility
echo "Checking GPU compatibility..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "ERROR: This project requires CUDA-compatible GPUs, which are not available on macOS."
    echo ""
    echo "Apple Silicon Macs use Metal, not CUDA. This project requires:"
    echo "  • NVIDIA GPUs with CUDA support"
    echo "  • Ampere, Ada, Hopper, or Turing architecture (A100, RTX 3090/4090, H100, T4, RTX 2080, etc.)"
    echo ""
    echo "Please run this on a system with compatible NVIDIA hardware."
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. This may indicate:"
    echo "   • No NVIDIA GPU present"
    echo "   • NVIDIA drivers not installed"
    echo "   • Running on incompatible hardware"
    echo ""
    echo "This project requires CUDA-compatible NVIDIA GPUs:"
    echo "  • Ampere, Ada, Hopper: A100, RTX 3090/4090, H100"
    echo "  • Turing: T4, RTX 2080"
    echo ""
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting. Please ensure you have compatible NVIDIA hardware."
        exit 1
    fi
elif ! nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA GPU not accessible or drivers not working properly."
    echo "Please check your NVIDIA driver installation."
    exit 1
else
    echo "NVIDIA GPU detected"
fi

# Try to pull the Docker image first, build as fallback
echo "Attempting to pull pre-built Docker image..."
IMAGE_NAME="kzkedzierska/sc_foundation_evals:latest_notebook"

if docker pull "$IMAGE_NAME" &> /dev/null; then
    echo "Using pre-built Docker image"
else
    echo "Pre-built image not available, building locally..."
    
    # Build base image
    pushd envs/docker/base_image || exit 1
    if ! docker build -t kzkedzierska/sc_foundation_evals:latest .; then
        echo "ERROR: Failed to build base Docker image"
        exit 1
    fi
    popd || exit 1
    
    # Build jupyter image with same tag as remote for consistency
    pushd envs/docker/jupyter || exit 1
    if ! docker build -t "$IMAGE_NAME" .; then
        echo "ERROR: Failed to build Jupyter Docker image"
        exit 1
    fi
    popd || exit 1
    
    echo "Local build completed successfully!"
fi

# Run Jupyter in Docker with volume mounting
echo "Starting Jupyter notebook in Docker..."
docker run -it --rm \
    --gpus all \
    -p 8888:8888 \
    -v "$(pwd)":/workspace \
    -v ~/.huggingface:/root/.huggingface \
    "$IMAGE_NAME"