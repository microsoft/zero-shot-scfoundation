# Development Setup

## Using Docker (Recommended)

The project includes HuggingFace models that are installed in Docker containers rather than pip requirements.

### Option 1: Run Jupyter with Docker script

```bash
./run_jupyter.sh
```

### Option 2: Use Docker Compose

```bash
# Start Jupyter notebook
docker-compose up jupyter

# Or start a development container
docker-compose up -d dev
docker-compose exec dev bash
```

### Option 3: Manual Docker commands

```bash
# Build images
cd envs/docker/base_image
docker build -t kzkedzierska/sc_foundation_evals:latest .
cd ../jupyter  
docker build -t sc_foundation_jupyter:latest .

# Run Jupyter
docker run -it --rm --gpus all -p 8888:8888 -v $(pwd):/workspace sc_foundation_jupyter:latest
```

## Local Installation (Not Recommended for HF Models)

If you need to install locally, you'll need to manually install the HuggingFace models:

```bash
# Install base requirements
pip install -r requirements.txt

# Install HF models (not included in requirements.txt per best practices)
pip install git+https://github.com/bowang-lab/scGPT.git@v0.1.6
pip install git+https://huggingface.co/ctheodoris/Geneformer.git@5d0082c1e188ab88997efa87891414fdc6e4f6ff
```

## Why Docker?

1. **HF Model Management**: HuggingFace models shouldn't be in requirements.txt per best practices
2. **Reproducibility**: Exact environment with proper CUDA/PyTorch versions
3. **Isolation**: No conflicts with your local Python environment
4. **GPU Support**: Properly configured CUDA environment
