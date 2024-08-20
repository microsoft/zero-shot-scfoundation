FROM cnstark/pytorch:1.13.0-py3.9.12-cuda11.7.1-ubuntu20.04

# NAME sc_foundation_evals

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y wget git git-lfs && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
  dpkg -i cuda-keyring_1.1-1_all.deb && \
  rm cuda-keyring_1.1-1_all.deb && \
  apt-get update && \
  echo "tzdata tzdata/Areas select Europe" > /tmp/prelseed.txt; \
  echo "tzdata tzdata/Zones/Europe select Warsaw" >> /tmp/preseed.txt; \
  debconf-set-selections /tmp/preseed.txt && \
  apt-get install -y cuda-toolkit-11-7 && \
  apt-get install -y r-base && \
  apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

RUN pip install packaging && \
  pip install flash-attn==1.0.4 --no-build-isolation 
  
RUN pip install scib[kBET,rpy2] colorlog PyComplexHeatmap wandb && \
  pip install git+https://github.com/bowang-lab/scGPT.git@v0.1.6 && \
  pip install git+https://huggingface.co/ctheodoris/Geneformer.git@5d0082c1e188ab88997efa87891414fdc6e4f6ff && \
  pip install git+https://github.com/microsoft/zero-shot-scfoundation.git@v0.1.0
