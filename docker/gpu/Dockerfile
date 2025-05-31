
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu20.04

ARG PY_VERSION=3.11

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
    software-properties-common \
    vim 

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
    python${PY_VERSION} \
    python${PY_VERSION}-distutils \
    python${PY_VERSION}-venv \
    python${PY_VERSION}-dev \
    python3-pip

RUN ln -sf /usr/bin/python${PY_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip
RUN ln -sf /usr/bin/python${PY_VERSION} /usr/bin/python3

ENV PATH="/root/.local/bin:$PATH"
COPY requirements.txt ./
RUN python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir -r requirements.txt
WORKDIR /workspace
COPY .. /workspace
CMD ["bash"]