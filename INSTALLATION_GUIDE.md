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

### Mounting Additional Data Directories

By default, only the project directory is mounted at `/workspace`. To access datasets or files outside the project:

#### Option 1: Modify run_jupyter.sh

```bash
# Edit run_jupyter.sh to add additional volume mounts
docker run -it --rm \
    --gpus all \
    -p 8888:8888 \
    -v "$(pwd)":/workspace \
    -v ~/.huggingface:/root/.huggingface \
    -v /path/to/your/datasets:/data \
    -v /path/to/additional/files:/mnt/files \
    sc_foundation_jupyter:latest
```

#### Option 2: Add to `docker-compose.yml`

```yaml
volumes:
  - .:/workspace
  - ~/.huggingface:/root/.huggingface
  - /path/to/your/datasets:/data
  - /path/to/additional/files:/mnt/files
```

#### Option 3: One-time manual run

```bash
docker run -it --rm --gpus all -p 8888:8888 \
  -v $(pwd):/workspace \
  -v /absolute/path/to/data:/data \
  sc_foundation_jupyter:latest
```

**Access mounted data in notebooks:**

- Project files: `/workspace/`
- Additional data: `/data/` (or whatever you mounted it as)
- Files are read/write accessible from within the container

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

1. **Reproducibility**: Exact environment with proper CUDA/PyTorch versions
2. **Isolation**: No conflicts with your local Python environment
3. **GPU Support**: Properly configured CUDA environment
