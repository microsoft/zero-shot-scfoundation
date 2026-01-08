# Foundation models in single-cell biology: evaluating zero-shot capabilities

[![DOI](https://badgen.net/badge/DOI/10.1101%2F2023.10.16.561085/red)](https://www.biorxiv.org/content/10.1101/2023.10.16.561085) [![DOI](https://badgen.net/badge/figshare/10.6084%2Fm9.figshare.24747228/green)](https://doi.org/10.6084/m9.figshare.24747228)

This repository contains the code that accompanies our paper, **Assessing the limits of zero-shot foundation models in single-cell biology**. You can find the preprint of the paper [here](https://www.biorxiv.org/content/10.1101/2023.10.16.561085).

## Project overview

In this project, we assess two proposed foundation models in the context of single-cell RNA-seq: Geneformer ([pub](https://www.nature.com/articles/s41586-023-06139-9), [code](https://huggingface.co/ctheodoris/Geneformer)) and scGPT ([pub](https://www.biorxiv.org/content/10.1101/2023.04.30.538439v2), [code](https://github.com/bowang-lab/scGPT)). We focus on evaluating the zero-shot capabilities of these models, specifically their ability to generalize beyond their original training objectives. Our evaluation targets two main tasks: cell type clustering and batch integration. In these tasks, we compare the performance of Geneformer and scGPT against two baselines: scVI  ([pub](https://www.nature.com/articles/s41592-018-0229-2), [code](https://docs.scvi-tools.org/en/stable/user_guide/models/scvi.html)) and a heuristic method that selects highly variable genes (HVGs). We also investigate the performence of the models in reconstructing the gene expression profiles of cells, and compare it against the baselines - such as a mean expression value or average ranking.

## Dependencies

This code has been developed and tested on Linux systems with NVIDIA GPUs.

**Compatible GPUs:**

- **Ampere, Ada, or Hopper**: A100, RTX 3090, RTX 4090, H100
- **Turing**: T4, RTX 2080

The code requires flash-attention for scGPT, which has strict GPU requirements.

<details>
<summary>Packages version</summary>

This code has been tested with the following versions of the packages:

- Python - tested with `3.9`
- PyTorch - tested with - `1.13`
- CUDA - tested with `11.7`
- [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/v1.0.4) - depends on `v1.0.4`
- [scGPT](https://github.com/bowang-lab/scGPT/tree/v0.1.6) - depends on `v0.1.6`
- [Geneformer](https://huggingface.co/ctheodoris/Geneformer/tree/5d0082c1e188ab88997efa87891414fdc6e4f6ff) - depends on commit `5d0082c`
- [scIB](https://github.com/theislab/scib/tree/v1.0.4) - tested with `v1.0.4`
- [sc_foundation_evals](https://github.com/microsoft/zero-shot-scfoundation) `v0.1.0`

</details>

## Quick Start (Docker - Recommended)

**Prerequisites:**

- **CUDA-compatible NVIDIA GPU** (see Dependencies section above)
- Docker with GPU support ([Installation Guide](https://docs.docker.com/get-docker/))
- NVIDIA Container Toolkit for GPU access ([Setup Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

**Note:** If you're using Docker in a non-privileged environment (clusters, shared systems), make sure your user is in the `docker` group or Docker is configured for rootless operation. See [Docker post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) for details.

```bash
# Clone repository and get data
git clone https://github.com/microsoft/zero-shot-scfoundation
cd zero-shot-scfoundation
wget https://figshare.com/ndownloader/files/43480497 -O data.zip
unzip data.zip && rm data.zip

# Start Jupyter notebooks
docker-compose up jupyter
# Or use the convenience script
./run_jupyter.sh
```

**Note:** This automatically pulls the pre-built Docker image `kzkedzierska/sc_foundation_evals:latest_notebook` from Docker Hub.

Open <http://localhost:8888> to access Jupyter notebooks.

**📋 For detailed installation options and troubleshooting, see [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)**

## Running the code

### Notebooks

To best understand the code and it's organization, please have a look at the notebooks. The `notebooks` directory currently contains the following notebooks:

- [scGPT_zero_shot](notebooks/scGPT_zero_shot.ipynb) - notebook for running scGPT zero-shot evaluation
- [Geneformer_zero_shot](notebooks/Geneformer_zero_shot.ipynb) - notebook for running Geneformer zero-shot evaluation
- [Baselines_HVG_and_scVI](notebooks/Baselines_HVG_and_scVI.ipynb) - notebook for running the baselines used in the paper, i.e. HVG and scVI.

## Any questions?

If you have any questions, or find any issues with the code, please open an issue in this repository. You can find more information on how to file an issue in [here](/SUPPORT.md). We also welcome any contributions to the code - be sure to checkout the **Contributing** section below.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
