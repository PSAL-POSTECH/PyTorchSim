import torch
import torch._dynamo
import torch.utils.cpp_extension

import argparse
import datetime


def run_layernorm(size, config):
    from Scheduler.scheduler import Scheduler, SchedulerDNNModel, Request
    scheduler = Scheduler(num_request_queue=1, engine_select=Scheduler.FIFO_ENGINE, togsim_config=config)
    device = scheduler.execution_engine.module.custom_device()
    input = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(torch.nn.LayerNorm(size[-1]).to(device=device))

    SchedulerDNNModel.register_model("LayerNorm", opt_fn)
    request = Request("LayerNorm", [input], [], request_queue_idx=0)
    scheduler.add_request(request, request_time=0)

    # Run scheduler
    while not scheduler.is_finished():
        scheduler.schedule()

    print(f"LayerNorm {str(size)} Simulation Done")

if __name__ == "__main__":
    import os
    import sys
    base_dir = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
    config = os.environ.get('TORCHSIM_CONFIG', default=f'{base_dir}/configs/systolic_ws_128x128_c2_simple_noc_tpuv4.yml')
    config_prefix = config.split('/')[-1].split('.')[0][9:] # extract config name from config path
    sys.path.append(base_dir)
    args = argparse.ArgumentParser()
    args.add_argument('--size', nargs='+', type=int, default=[512, 768], help='Tensor Shape')
    args.add_argument('--dump_path', type=str, default='results')
    args = args.parse_args()
    size = args.size
    size_str = "x".join([str(i) for i in size])
    result_path = os.path.join(base_dir, args.dump_path, config_prefix, f"LayerNorm_{size_str}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    # setting environment variables
    os.environ['TORCHSIM_LOG_PATH'] = result_path
    os.environ['TORCHSIM_FUSION_REDUCTION_REDUCTION'] = "0"
    # only timing simulation
    os.environ['TORCHSIM_VALIDATION_MODE'] = "0"
    if 'pytorchsim_functional_mode' in os.environ:
        del os.environ['pytorchsim_functional_mode']

    run_layernorm(size, config)
