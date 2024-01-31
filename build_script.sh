#!bin/sh

# Gemmini Lowering Pass build
cd GemminiLowerPass
mkdir -p build
cd build
cmake ..
make -j
cd ../..

# checkout to TorchSim branch
cd ONNXim
git checkout TorchSim
git submodule update --recursive --init

# ONNXim build
mkdir -p build
cd build
conan install ..
conan install .. --build=missing
cmake install ..
make -j
cd ..