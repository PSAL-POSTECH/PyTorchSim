# Simulation components

This directory contains the host-side Spike source and local simulator artifacts used by PyTorchSim.

- `riscv-isa-sim/`: independent Git checkout of the patched Spike source.
- `riscv-isa-sim-install/`: local Spike installation produced from that checkout.
- `spike-build-tools/`: temporary host build helper tools.

Build the simulator from the nested checkout, then select the installed binary explicitly when running PyTorchSim. The PyTorchSim repository does not automatically invoke the nested checkout.

```bash
cd sim/riscv-isa-sim
mkdir -p build && cd build
../configure --prefix="$PWD/../../riscv-isa-sim-install"
make -j"$(nproc)"
make install
```
