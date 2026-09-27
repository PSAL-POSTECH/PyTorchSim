#!/bin/bash
home="/workspace"
cd $home

# Gem5
apt -y update && apt -y upgrade && apt -y install scons
git clone https://github.com/PSAL-POSTECH/gem5.git
cd gem5 && scons build/RISCV/gem5.opt -j $(nproc)
export GEM5_PATH=$home/gem5/build/RISCV/gem5.opt
cd $home

# LLVM
git clone https://github.com/PSAL-POSTECH/llvm-project.git --branch ispass2026
cd llvm-project && git apply $home/PyTorchSim/scripts/patches/llvm-project-pytorchsim-internal-accumulator.patch && mkdir build && cd build && \
  cmake -DLLVM_ENABLE_PROJECTS=mlir -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/riscv-llvm -DLLVM_TARGETS_TO_BUILD=RISCV -G "Unix Makefiles" ../llvm && \
  make -j && make install
cd $home

# Spike Simulator
git clone https://github.com/PSAL-POSTECH/riscv-isa-sim.git --branch ispass2026 && cd riscv-isa-sim && \
    git apply $home/PyTorchSim/scripts/patches/riscv-isa-sim-internal-accumulator.patch && \
    mkdir build && cd build && \
    ../configure --prefix=$RISCV && make -j && make install
cd $home
