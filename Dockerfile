# syntax=docker/dockerfile:1.4
ARG BASE_IMAGE=ghcr.io/psal-postech/torchsim_base:latest
FROM ${BASE_IMAGE}

# Prepare PyTorchSim project
COPY . /workspace/PyTorchSim

RUN cd PyTorchSim/TOGSim && \
    mkdir -p build && \
    cd build && \
    conan install .. --build=missing && \
    cmake .. && \
    make -j$(nproc)

RUN cd PyTorchSim/PyTorchSimDevice && \
    python -m pip install --no-build-isolation -e .