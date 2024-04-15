# Copyright (c) 2020 The Regents of the University of California
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Copied from Gem5 Docker file
ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update && apt -y upgrade && \
    apt -y install build-essential git m4 scons zlib1g zlib1g-dev \
    libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
    python3-dev python-is-python3 doxygen libboost-all-dev \
    libhdf5-serial-dev python3-pydot libpng-dev libelf-dev pkg-config pip \
    python3-venv black
RUN pip install mypy pre-commit

# Build Gem5
ENV LD_LIBRARY_PATH /opt/conda/lib:$LD_LIBRARY_PATH
RUN git clone https://github.com/PSAL-POSTECH/gem5.git --branch TorchSim
RUN cd gem5 && scons build/RISCV/gem5.opt -j $(nproc)

# Build LLVM RISC-V
RUN git clone https://github.com/llvm/llvm-project.git
RUN cd llvm-project && git checkout release/17.x && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/riscv-llvm -DLLVM_TARGETS_TO_BUILD=RISCV -G "Unix Makefiles" ../llvm && \
    make -j && make install

# Store RISC-V LLVM for TorchSim
ENV TORCHSIM_LLVM_PATH /riscv-llvm/bin
ENV TORCHSIM_LLVM_INCLUDE_PATH /riscv-llvm/include
ENV LLVM_DIR /riscv-llvm

# Download RISC-V tool chain
RUN apt install -y wget && \
    wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2023.12.14/riscv64-glibc-ubuntu-22.04-llvm-nightly-2023.12.14-nightly.tar.gz && \
    wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2023.12.14/riscv64-elf-ubuntu-20.04-llvm-nightly-2023.12.14-nightly.tar.gz && \
    tar -zxvf riscv64-elf-ubuntu-20.04-llvm-nightly-2023.12.14-nightly.tar.gz && tar -zxvf riscv64-elf-ubuntu-20.04-llvm-nightly-2023.12.14-nightly.tar.gz && \
    rm *.tar.gz 

ENV RISCV /workspace/riscv
ENV PATH $RISCV/bin:$PATH

# Install Spike simulator
RUN apt -y install device-tree-compiler
RUN git clone https://github.com/PSAL-POSTECH/riscv-isa-sim.git --branch TorchSim && cd riscv-isa-sim && mkdir build && cd build && \
    ../configure --prefix=$RISCV && make -j && make install

# Install Proxy kernel
RUN git clone https://github.com/riscv-software-src/riscv-pk.git && cd riscv-pk && mkdir build && cd build && \
    ../configure --prefix=$RISCV --host=riscv64-unknown-elf && make -j && make install

# Install torchsim dependency
RUN apt install ninja-build && pip install onnx && pip install --user conan==1.56.0
ENV PATH $PATH:/root/.local/bin
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
