ARG BASE_IMAGE=ghcr.io/psal-postech/torchsim_base:latest
FROM ${BASE_IMAGE}

# The base is the compiler's image, which carries TOGSim built (third_party/TOGSim): the header a
# TOG is compiled against and the Simulator that runs it come from there, and nothing of TOGSim
# is built here. This image adds the PyTorchSim under test and its device extension.
COPY . /workspace/PyTorchSim

RUN cd PyTorchSim/device && \
    python -m pip install --no-build-isolation -e .
