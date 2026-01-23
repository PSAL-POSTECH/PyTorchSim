import torch
import torch._dynamo
import torch.utils.cpp_extension

import argparse
import datetime


def run_matmul(input_size, hidden_size, output_size, config):
    from Scheduler.scheduler import Scheduler, SchedulerDNNModel, Request
    def custom_matmul(a, b):
        return torch.matmul(a, b)
    scheduler = Scheduler(num_request_queue=1, engine_select=Scheduler.FIFO_ENGINE, togsim_config=config)
    device = scheduler.execution_engine.module.custom_device()
    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size).to(device=device)
    weight = torch.randn(hidden_size, output_size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(custom_matmul)

    SchedulerDNNModel.register_model("GEMM", opt_fn)
    request = Request("GEMM", [input, weight], [], request_queue_idx=0)
    scheduler.add_request(request, request_time=0)

    # Run scheduler
    while not scheduler.is_finished():
        scheduler.schedule()

    print(f"GEMM {input_size}x{hidden_size}x{output_size} (MxKxN) Simulation Done")

if __name__ == "__main__":
    import os
    import sys
    base_dir = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
    config = os.environ.get('TORCHSIM_CONFIG', default=f'{base_dir}/configs/systolic_ws_128x128_c2_simple_noc_tpuv4.yml')
    config_prefix = config.split('/')[-1].split('.')[0][9:] # extract config name from config path
    sys.path.append(base_dir)
    args = argparse.ArgumentParser()
    args.add_argument('--size', nargs='+', type=int, default=[128, 128, 128], help='M K N')
    args.add_argument('--dump_path', type=str, default='results')
    args = args.parse_args()
    size = args.size
    size_str = "x".join([str(i) for i in size])
    result_path = os.path.join(base_dir, args.dump_path, config_prefix, f"GEMM_{size_str}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    # setting environment variables
    os.environ['TORCHSIM_LOG_PATH'] = result_path
    # only timing simulation
    os.environ['TORCHSIM_VALIDATION_MODE'] = "0"
    if 'pytorchsim_functional_mode' in os.environ:
        del os.environ['pytorchsim_functional_mode']

    run_matmul(size[0], size[1], size[2], config)
