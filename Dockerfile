ARG BASE_IMAGE=ghcr.io/psal-postech/torchsim_base:latest
FROM ${BASE_IMAGE}

# The compiler image carries the PyTorchSim it pins, for its own TOGSim; this image is about the
# PyTorchSim under test, so that checkout goes and the tree being tested takes its place, TOGSim
# included (the compiler reads TOGSim's headers from here when it builds a trace producer).
RUN rm -rf /workspace/PyTorchSim
COPY . /workspace/PyTorchSim

RUN cd PyTorchSim/TOGSim && \
    mkdir -p build && \
    cd build && \
    conan install .. --build=missing && \
    cmake .. && \
    make -j$(nproc)

RUN cd PyTorchSim/PyTorchSimDevice && \
    python -m pip install --no-build-isolation -e .
