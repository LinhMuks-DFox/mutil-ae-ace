
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ARG PY_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    git \
    cmake \
    build-essential \
    sox \
    libsox-dev \
    libsndfile1 \
    llvm \
    clangd \
    clang-format \
    tzdata \
    software-properties-common 

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
    python${PY_VERSION} \
    python${PY_VERSION}-distutils \
    python${PY_VERSION}-venv \
    python${PY_VERSION}-dev \
    python3-pip

RUN ln -sf /usr/bin/python${PY_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip
RUN pip install nvitop
ENV PATH="/root/.local/bin:$PATH"
COPY requirements.txt ./
RUN pip install -r requirements.txt
WORKDIR /workspace
COPY .. /workspace
CMD ["bash"]