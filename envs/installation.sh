#! /bin/bash
# exit on error
set -e

_script_name=$(basename "$0")

ENV_NAME="sc_foundation_evals"

warning() {
  yellow='\033[0;33m'
  nc='\033[0m'
  echo -e "${yellow}$(date '+%Y-%m-%d %H:%M:%S') WARNING: $@${nc}" 1>&2
}

success() {
  green='\033[0;32m'
  nc='\033[0m'
  echo -e "${green}$(date '+%Y-%m-%d %H:%M:%S') SUCCESS: $@${nc}"
}

error() {
  red='\033[0;31m'
  nc='\033[0m'
  echo -e "${red}$(date '+%Y-%m-%d %H:%M:%S') ERROR: $@${nc}" 1>&2
  usage_and_exit 1
}

msg() {
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') INFO: $@"
}

usage() {
  echo -e "

USAGE: bash ${_script_name}

Script to install the package and set up the Conda environment.

EXAMPLES:
  Install the package and set up the Conda environment:
    bash ${_script_name}
  "
}

usage_and_exit() {
  usage
  exit $1

}

# if mamba available, use it
if command -v mamba &>/dev/null; then
  conda_cli=mamba
else
  conda_cli=conda
fi
msg "Using '${conda_cli}' as the Conda CLI."

${conda_cli} env create -f envs/conda_env.yml -n ${ENV_NAME} ||
  error "Failed to create the Conda environment '${ENV_NAME}'."
success "Conda environment '${ENV_NAME}' created successfully."

${conda_cli} run \
  -n ${ENV_NAME} pip install flash-attn==1.0.4 --no-build-isolation
success "Flash attention installed successfully."

${conda_cli} run \
  -n ${ENV_NAME} pip install 'setuptools>=65.2' wandb colorlog \
  PyComplexHeatmap scib[kBET,rpy2]==1.0.4 ||
  error "Failed to install the wandb, colorlog, PyComplexHeatmap or scib."

${conda_cli} run \
  -n ${ENV_NAME} pip install git+https://github.com/bowang-lab/scGPT.git@v0.1.6 ||
  error "Failed to install the scGPT."

${conda_cli} run \
  -n ${ENV_NAME} pip install \
  git+https://huggingface.co/ctheodoris/Geneformer.git@5d0082c1e188ab88997efa87891414fdc6e4f6ff ||
  error "Failed to install the Geneformer."

${conda_cli} run \
  -n ${ENV_NAME} pip install git+https://github.com/microsoft/zero-shot-scfoundation ||
  error "Failed to install the sc_foundation_evals."

success "All packages installed successfully."
