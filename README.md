# PyTorchSim: A Comprehensive, Fast, and Accurate NPU Simulation Framework
[![Docker Image CI](https://github.com/PSAL-POSTECH/PyTorchSim/actions/workflows/docker-image.yml/badge.svg)](https://github.com/PSAL-POSTECH/PyTorchSim/actions/workflows/docker-image.yml)

PyTorchSim is a comprehensive, high-speed, cycle-accurate NPU simulation framework.
- We define a RISC-V-based NPU architecture and implement PyTorch compiler backend to run inference & training for PyTorch models.
- Achieved high speed and accuracy with our novel Tile-Level Simulation (TLS) with compiler-generated Tile-Operation Graph (TOG), exploiting deterministic tile compute latency.
- A generic and extensible NPU architecture based on RISC-V vector extension.
- The functional simulator supports code correctness validation and data-dependent timing simulation.


For more details, please refer to our [paper](https://doi.org/10.1145/3725843.3756045)!

## Navigation
[Overview](#pytorchsim-framework-overview) | [Model Zoo](#model-zoo) | [Getting Started](#getting-started)

<!-- ![Speedup](/docs/speedup.jpg)
**Figure description**: we compare the simulation speed of PyTorchSim over [Accel-sim](https://accel-sim.github.io/) (a GPU simulator with Tensor Core model) as GPUs are widely used for deep learning and such a GPU simulator can be used to study systems for deep learning. We also include [mNPUsim](https://github.com/casys-kaist/mNPUsim) in the comparison. On the x-axis, we vary the size and workloads.

![Validation](/docs/validation.jpg)
**Figure description**: PyTorchSim achieves significantly better accuracy than others ([SCALE-Simv3](https://github.com/scalesim-project/scale-sim-v3), [mNPUsim](https://github.com/casys-kaist/mNPUsim), [Timeloop](https://github.com/NVlabs/timeloop), [MAESTRO](https://github.com/maestro-project/maestro)) by supporting various optimizations, data transformations, and general vector operations. It achieved an 11.5% MAE (Mean Absolute Error) of runtime relative to the real TPUv3. -->

## PyTorchSim Framework Overview
![Overview](/docs/overview.jpg)
PyTorchSim consists of **two main** components:
- **Compiler**: Integrated of [PyTorch2](https://github.com/pytorch/pytorch) compiler stack and generates NPU machine code and TOG for existing PyTorch models.
- **TOGSim**: Executes TOG for high-speed simulation and accurately models shared resources (DRAM, NoC) through integrated cycle-accurate simulators ([BookSim](https://github.com/booksim/booksim2) and [Ramulator2](https://github.com/CMU-SAFARI/ramulator2)).

PyTorchSim **supports**:
- DNN inference and [training](#training)
- Data-dependent timing modeling (e.g. sparsity)
- [Multi-tenancy](#multi-tenancy)
- [Compiler optimizations](#compiler-optimizations)
- [Mapping](#mapping)
- [L2 Cache](#l2-cache) (persistent cache)

## Model Zoo
| Model | Source | Status | Note |
|---|:-:|:-:|---|
| ResNet-18 | <img src="https://avatars.githubusercontent.com/u/21003710?s=48&v=4" width="20"/> | ✅ | channel last format |
| ResNet-50 | <img src="https://avatars.githubusercontent.com/u/21003710?s=48&v=4" width="20"/> | ✅ | channel last format |
| BERT | <img src="https://avatars.githubusercontent.com/u/21003710?s=48&v=4" width="20"/> | ✅ |  |
| GPT-2 | <img src="https://avatars.githubusercontent.com/u/21003710?s=48&v=4" width="20"/> | ✅ |  |
| ViT | <img src="https://avatars.githubusercontent.com/u/21003710?s=48&v=4" width="20"/> | ✅ |  |
| Mistral | <img src="https://avatars.githubusercontent.com/u/21003710?s=48&v=4" width="20"/> | ✅ | |
| Diffusion | 🤗 | ✅ |  |
| Llama-4 | 🤗 | ⏳ | Under Development |
| DeepSeek v1 | 🤗 | ⏳ | Under Development |
<!-- ## Requirements

### OS Distribution
Recommended: Ubuntu 22.04

### Tested Environment
```bash
gcc == 11.4.0
g++ == 11.4.0
cmake == 3.26.4
conan == 1.56.0
python >= 3.10
pytorch == 2.2.0
risc-v64-unknown-elf-gcc == 13.2.0
```
Our provided Docker environment resolves software dependencies.

### Hardware Dependencies
Any x86 hardware capable of running Docker with more than 20 GB of memory -->

## Supported Operations
- GEMM
- Batched GEMM
- Convolution
- Elementwise
- Reduction
- Batchnorm
- Layernorm
- Softmax
- Transpose
- View
- Activation
- Pooling
- Etc (WIP)

## Getting Started
### Quick start with pre-built Docker image

To download the latest Docker image and set up the environment, use the following commands:

```bash
# Run the Docker container
docker run -it --ipc=host --name torchsim -w /workspace/PyTorchSim ghcr.io/psal-postech/torchsim-ci:v1.0.1 bash
```
### Manual Setting (Optional)
This script provides building [Gem5](https://github.com/PSAL-POSTECH/gem5.git), [LLVM](https://github.com/PSAL-POSTECH/llvm-project.git), and [Spike](https://github.com/PSAL-POSTECH/riscv-isa-sim.git) simulator from source code for specific experts.
```bash
bash script/build_from_source.sh
```
### Run Examples
The `tests` directory contains several AI workloads examples.
```bash
python tests/test_matmul.py 
```
The result is stored to `TORCHSIM_LOG_PATH/hash/togsim_result/`. The log file contains detailed core, memory, and interconnect stats.

### Run Your Own Model on PyTorchSim
You can run your own PyTorch model on PyTorchSim by setting up a custom NPU device.  
This method also applies when you want to simulate models beyond the provided examples.
```python
import torch
from Scheduler.scheduler import PyTorchSimRunner
# Declare a custom NPU device
device = PyTorchSimRunner.setup_device().custom_device()

# Declare you own model (e.g. resnet18 from torchvision)
from torchvision.models import resnet18
model = resnet18().eval()
x = torch.randn(1, 3, 224, 224, dtype=torch.float32)

# Move model and input tensors to the custom device
model.to(device)
x = x.to(device)

# Compile and run the model with PyTorchSim
compiled_model = torch.compile(dynamic=False)(model)
y = compiled_model(x)
```
`model` is your PyTorch model to be simulated, and `x` is the input tensor.
PyTorchSim automatically generates a Tile-Operation Graph (TOG), and runs it through the TOGSim backend.

### Result
Running log in CLI
```bash
Wrapper Codegen Path = /tmp/torchinductor_root/fo/cfofsp5nwmpqxctouan2v2t5y7qp5vwrgvw4swssx4ca4us3c5tx.py
[Gem5] Gem5 is running.
[Spike] Running Spike simulator
[TOGSim] TOGSim is running..
[TOGSim] Simulation log is stored to "/workspace/PyTorchSim/togsim_results/20251205_080553.log"
----------------------------
|Matmul Forward Test Passed|
----------------------------
```

Simulation consists of three steps

1. `Gem5` obatins compute latency for TOG.
2. `Spike` verifies the output code.
3. `TOGSim` simulates a NPU architecture.

If you want to turn off the `SpikeSimulator` for fast simulation, you can set as below.
```bash
export pytorchsim_functional_mode=False
```
Log contains memory & core stats.
```bash
[2025-12-05 08:05:52.538] [info] HBM2-CH_0: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 956, Row misses: 32, Row conflicts: 36
[2025-12-05 08:05:52.538] [info] HBM2-CH_1: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 956, Row misses: 32, Row conflicts: 36
[2025-12-05 08:05:52.538] [info] HBM2-CH_2: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 959, Row misses: 32, Row conflicts: 33
[2025-12-05 08:05:52.538] [info] HBM2-CH_3: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 956, Row misses: 32, Row conflicts: 36
[2025-12-05 08:05:52.538] [info] HBM2-CH_4: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 959, Row misses: 32, Row conflicts: 33
[2025-12-05 08:05:52.538] [info] HBM2-CH_5: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 959, Row misses: 32, Row conflicts: 33
[2025-12-05 08:05:52.538] [info] HBM2-CH_6: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 956, Row misses: 32, Row conflicts: 36
[2025-12-05 08:05:52.538] [info] HBM2-CH_7: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 958, Row misses: 32, Row conflicts: 34
[2025-12-05 08:05:52.538] [info] HBM2-CH_8: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 959, Row misses: 32, Row conflicts: 33
[2025-12-05 08:05:52.538] [info] HBM2-CH_9: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 959, Row misses: 32, Row conflicts: 33
[2025-12-05 08:05:52.538] [info] HBM2-CH_10: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 958, Row misses: 32, Row conflicts: 34
[2025-12-05 08:05:52.538] [info] HBM2-CH_11: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 959, Row misses: 32, Row conflicts: 33
[2025-12-05 08:05:52.538] [info] HBM2-CH_12: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 958, Row misses: 32, Row conflicts: 34
[2025-12-05 08:05:52.538] [info] HBM2-CH_13: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 958, Row misses: 32, Row conflicts: 34
[2025-12-05 08:05:52.538] [info] HBM2-CH_14: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] Row hits: 959, Row misses: 32, Row conflicts: 33
[2025-12-05 08:05:52.538] [info] HBM2-CH_15: avg BW utilization 49% (768 reads, 256 writes)
[2025-12-05 08:05:52.538] [info] ===== Instructions count =====
[2025-12-05 08:05:52.538] [info] Core [0] : MOVIN    inst_count 3
[2025-12-05 08:05:52.538] [info] Core [0] : MOVOUT   inst_count 1
[2025-12-05 08:05:52.538] [info] Core [0] : COMP     inst_count 10 (GEMM: 8, Vector: 2)
[2025-12-05 08:05:52.538] [info] Core [0] : BAR      inst_count 8
[2025-12-05 08:05:52.538] [info] ========= Core stat =========
[2025-12-05 08:05:52.538] [info] Core [0] : Systolic array [0] utilization(%) 12.40, active_cycles 256, idle_cycles 1809
[2025-12-05 08:05:52.538] [info] Core [0] : Systolic array [1] utilization(%) 12.40, active_cycles 256, idle_cycles 1809
[2025-12-05 08:05:52.538] [info] Core [0] : DMA active_cycles, 1024 DMA idle_cycles 1041, DRAM BW 238.000 GB/s (16384 responses)
[2025-12-05 08:05:52.538] [info] Core [0] : Vector unit utilization(%) 2.42, active cycle 50, idle_cycle 0
[2025-12-05 08:05:52.538] [info] Core [0] : NUMA local memory: 16384 requests, remote memory: 0 requests
[2025-12-05 08:05:52.538] [info] Core [0] : Total_cycles 2065
[2025-12-05 08:05:52.538] [info] Total execution cycles: 2065
[2025-12-05 08:05:52.538] [info] Wall-clock time for simulation: 0.147463 seconds
```
The log is dumped in `TORCHSIM_LOG_PATH` and you can set the path as below.
```bash
export TORCHSIM_LOG_PATH=/tmp/torchinductor # output file dump path
```

## Training
`backward()` automatically generates TOG and executes simulation for backward propagation. If you want to simulate optimizers on NPU units, you can compile the optimizer’s step function.
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
compiled_step = torch.compile(dynamic=False)(optimizer.step)

optimizer.zero_grad()
loss.backward()
opt_step()
```
`tests/test_mlp.py` provides an example of MLP training.

## Multi-tenancy
Our load generator supports multi-tenancy experiments. You can run a simple example by executing `tests/test_scheduler.py`.
```bash
python tests/test_scheduler.py
```
Below is an example code of multi-tenancy `resnet18` and `EncoderBlock`.
In this example, the `Scheduler` is initialized with a number of request queues, a scheduling policy, and a TOGSimulator config file(`.yml`). The compiled PyTorch models are then registered with a unique model id.

```python3
import os
import sys
import torch
from torchvision.models import resnet18
base_path = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
config = f'{base_path}/configs/systolic_ws_128x128_c2_simple_noc_tpuv3_partition.yml'

sys.path.append(base_path)
from tests.test_transformer import EncoderBlock
from Scheduler.scheduler import Scheduler, SchedulerDNNModel, Request, poisson_request_generator
scheduler = Scheduler(num_request_queue=2, engine_select=Scheduler.FIFO_ENGINE, togsim_config=config)

# Register compiled model
target_model0 = resnet18().eval()
target_model1 = EncoderBlock(768, 12).eval()
opt_model0 = torch.compile(target_model0.to(device=scheduler.execution_engine.module.custom_device(), memory_format=torch.channels_last))
opt_model1 = torch.compile(target_model1.to(device=scheduler.execution_engine.module.custom_device()))
SchedulerDNNModel.register_model("model0", opt_model0)
SchedulerDNNModel.register_model("model1", opt_model1)
```

The config file(`.yml`) specifies two key items:
- `num_partition`: The total number of independent request queues to create.
- `partition`: Defines the hardware mapping, assigning each queue (identified by its index) to a specific physical core.
For example, the configuration below creates two scheduling queues (`0` and `1`) and maps `core_0` to queue `0` and `core_1` to queue `1`:
```
  "num_partition" : 2,
  "partition": {
    "core_0":0,
    "core_1":1
  }
```

Next, DNN model requests are generated and submitted. We provide a `poisson_request_generator` utility, which generates request arrival times.
Each `Request` is created with its model name, data, and a request_queue_idx to specify its target queue, then added via `scheduler.add_request`.
As shown in the code, `model0` requests are queued to `request_queue_idx=0`, while `model1` requests are queued to `request_queue_idx=1`.
```python3
# Load Generation
model0_lambda = 5.0
model1_lambda = 3.0
max_time = 1000.0 # [s]

# Generate Possion distribution requests for model0
for model0_request_time in poisson_request_generator(model0_lambda, max_msec_time=max_time):
    x = torch.randn(1, 3, 224, 224)
    new_request = Request("model0", [x], [], request_queue_idx=0)
    scheduler.add_request(new_request, request_time=model0_request_time)

# Generate Possion distribution requests for model1
for model1_request_time in poisson_request_generator(model1_lambda, max_msec_time=max_time):
    x = torch.randn(128, 768)
    new_request = Request("model1", [x], [], request_queue_idx=1)
    scheduler.add_request(new_request, request_time=model1_request_time)
```

Finally, `scheduler.schedule()` is called in a loop until all requests are processed.
```python3
# Run scheduler
while not scheduler.is_finished():
    scheduler.schedule()
```

## Compiler Optimizations
PyTorchSim compiler supports several fusion optimizations:
- GEMM prologue fusion
- GEMM epilogue fusion
- GEMM reduction fusion
- CONV epilogue fusion

Depending on tensor shape, use different convolution template:
- Single batch optimization
- Multi-channel optimization

## Mapping
PyTorchSim provides three mapping strategies.
### Heuristic-based mapping
We adopt and modified heuristic-based mapping of [GEMMINI](https://github.com/ucb-bar/gemmini) by default, which maximizes the utilization of scratchpad memory.
### Auto-tuning
Heuristic method may not be optimal for all cases. PyTorchSim provides auto-tuning to find the best mapping for GEMM, CONV, and vector operations. It reduces the search space by sorting candidates based on scratchpad memory utilization and picking the top-k candidates. Search parameters include tile shape and vector lane stride.

To enable this, update your configuration file as follows:
```bash
"codegen_mapping_strategy" : "autotune"
```
### Manunal setting
Users can utilizing third-party mapping tools (e.g., Timeloop). You can explicitly set the mapping file path in the configuration file to apply your own mapping strategies.
```bash
"codegen_mapping_strategy" : "external",
"codegen_external_mapping_file" : "path/to/mapping_file.json",
```
Key: "M_N_K" for GEMM
```
{
    "512_2048_8192" : {
        "TILE_M" : 512,
        "TILE_K" : 512,
        "TILE_N" : 1024
    },
    "512_2048_2048" : {
        "TILE_M" : 512,
        "TILE_K" : 512,
        "TILE_N" : 1024
    },
    "2048_2048_512" : {
        "TILE_M" : 1024,
        "TILE_K" : 512,
        "TILE_N" : 512
    }
}
```

## L2 Cache
It supports L2 cache as persistent cache. User can provide software-managed allocation/eviction strategy for tensors with persistent cache.

Common Memory (CMEM) is a new feature introduced in the latest TPUs (newer than TPUv3). Multiple cores share this memory, which provides high bandwidth. Reusable tensors are stored and loaded from CMEM to avoid off-chip traffic. Our L2 cache can work like as CMEM

To allocate a tensor in L2 cache, set the environment variable as shown below. The `tpuv4` directory provides example plans for L2 cache obtained from TPUv4 profiling.
```bash
export SRAM_BUFFER_PLAN_PATH=tpuv4/gemm_plan.py
```
The L2 cache strategy file is composed as follows:
```
plan = {
    "arg0_1"
}
```
In this example, only one input tensor is registered in L2 cache. You can refer to the tensor name from the wrapper code. After running the code, you can find the wrapper codegen path in the [result](#result) section.

Last but not least, you must set `l2d_type` and `l2d_config` in the [TOGSim config](#togsim-configuration) to use L2 cache. The `l2d_config` follows the same configuration method as [AccelSim](https://github.com/accel-sim/accel-sim-framework).

## Compiler Configuration
`PyTorchSimFrontend/extension_config.py` contains target hardware configuration to compile.

You can configure these options using environment variables.
```bash
export TORCHSIM_DIR=/workspace/PyTorchSim # home directory

# Plan which tensor allocated in TPUv4's CMEM
export SRAM_BUFFER_PLAN_PATH=/workspace/PyTorchSim/tpuv4/gemm_plan.py

export TORCHSIM_TLS_MODE=1 # User can choose TLS or ILS mode
export TORCHSIM_USE_TIMING_POOLING=0 # use lightweight pooling for timing
```
## TOGSim Configuration
![NPU_Core](./docs/npu_core.jpg)

`configs` directory contains example NPU configuration files in the JSON format.
```
  "num_cores" : 2,                   // Number of NPU cores
  "core_freq_mhz" : 940,             // Core's frequency (MHz)
  "num_systolic_array_per_core" : 2, // Number of systolic array per core

  "vpu_num_lanes" : 128,             // Number of VPU lanes
  "vpu_spad_size_kb_per_lane" : 128, // Scratchpad memory size per lane (KB)
  "vpu_vector_length_bits" : 256,    // VPU vector register length (Bits)

  "dram_type" : "ramulator2",        // DRAM type (ex. ramulator2, simple)
  "dram_freq_mhz" : 940,             // DRAM frequency (MHz)
  "dram_channels": 32,               // Number of DRAM channels
  "dram_req_size": 32,               // DRAM request size (B)
  "dram_latency" : 10,               // DRAM latency (cycle)
  "dram_nbl" : 2,                    // DRAM burst length size
  "dram_config_path" : "../configs/ramulator2_configs/HBM2_TPUv3.yaml", // Ramulator2 config file path

  "l2d_type" : "datacache",
  "l2d_config" : "S:64:128:512,32,L:B:m:W:L,A:192:4,32:0,32",

  "icnt_type" : "simple",              // Interconnect type (ex. booksim, simple)
  "icnt_latency" : 7,                  // Interconnect latency (cycle)
  "icnt_freq_mhz" : 940,               // Interconnect frequency (MHz)
  "icnt_injection_ports_per_core" : 16 // Interconnect injection ports per core
  "icnt_config_path" : "../configs/booksim2_configs/fly_c4_m32.icnt", // Booksim2 config file path

  "precision" : 4,                   // Element's precision in tensor (Byte)
  "scheduler" : "simple",            // Scheduler type (Now, only support simple scheduler)
  "num_partition" : 2,               // Multi-core Partitioning
  "partition": {                     // allocate request queue index
    "core_0":0,
    "core_1":1
  },

  "codegen_mapping_strategy" : "heuristic", // Compiler mapping strategy (ex. "heuristic", "autotune", "external-then-heuristic", "external-then-autotune")
  "codegen_external_mapping_file" : "",     // Path to external mapping file
  "codegen_autotune_max_retry": 10,         // Maximum retries for autotuning
  "codegen_autotune_template_topk": 4,      // Top-K templates to consider during autotuning
  // Compiler optimization level/options.
  // Value can be "all", "none", or a list of specific optimizations:
  // ["fusion", "reduction_epilogue", "reduction_reduction", "prologue", "single_batch_conv", "multi_tile_conv", "subtile"]
  "codegen_compiler_optimization" : "all"
```
You can set TOGSim config path as below.
```bash
export TORCHSIM_CONFIG=/workspace/PyTorchSim/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.yml
```
## Future Works
Currently, PyTorchSim supports PyTorch 2.2. Support for newer versions will be added soon.

## Artifact Evaluation
Artifact evaluation is being prepared for v1.0.0.
The following scripts reproduce the validation and speedup results from the paper.
### Build
```bash
docker run -it --ipc=host --name torchsim -w /workspace/PyTorchSim ghcr.io/psal-postech/torchsim-ci:v1.0.0 bash
```

To generate validation results
```bash
# Run a cycle accuracy script
./experiments/artifact/cycle_validation/run_cycle.sh
```
To generate speedup results
```bash
# Run a speedup accuracy script
./experiments/artifact/speedup/run_speedup.sh
```

## Contributing
We welcome any contributions and issue reports.
Please refer to the [Contributing Guide](https://github.com/PSAL-POSTECH/PyTorchSim?tab=contributing-ov-file) for details.

## Citation
If you use PyTorchSim for your research, please cite the following paper.
```
@INPROCEEDINGS{yang2025pytorchsim,
  author={Yang, Wonhyuk and Shin, Yunseon and Woo, Okkyun and Park, Geonwoo and Ham, Hyungkyu and Kang, Jeehoon and Park, Jongse and Kim, Gwangsun},
  title={PyTorchSim: A Comprehensive, Fast, and Accurate NPU Simulation Framework},
  booktitle={Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture},
  pages={1363–1380},
  year={2025},
  doi={10.1145/3725843.3756045},
  series={MICRO '25}
}
```
