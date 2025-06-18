import torch
import torch._dynamo
import torch.utils.cpp_extension

import argparse
import datetime


def run_attention(size, config):
    def attention(query, key, value):
        import math
        d_k = query.size(-1)
        scores = torch.matmul(key, query.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = scores.softmax(dim=-2)
        return torch.matmul(value.transpose(-1, -2), p_attn)
    from Scheduler.scheduler import Scheduler, SchedulerDNNModel, Request
    scheduler = Scheduler(num_request_queue=1, engine_select=Scheduler.FIFO_ENGINE, backend_config=config)
    device = scheduler.execution_engine.module.custom_device()
    query = torch.randn(size).to(device=device)
    key = torch.randn(size).to(device=device)
    value = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(attention)

    SchedulerDNNModel.register_model("attention", opt_fn)
    request = Request("attention", [query, key, value], [], request_queue_idx=0)
    scheduler.add_request(request, request_time=0)

    # Run scheduler
    while not scheduler.is_finished():
        with torch.no_grad():
            scheduler.schedule()

    print(f"Attention {str(size)} Simulation Done")

if __name__ == "__main__":
    import os
    import sys
    base_dir = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
    config = os.environ.get('TORCHSIM_CONFIG', default=f'{base_dir}/PyTorchSimBackend/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.json')
    config_prefix = config.split('/')[-1].split('.')[0][9:] # extract config name from config path
    sys.path.append(base_dir)
    args = argparse.ArgumentParser()
    args.add_argument('--size', nargs='+', type=int, default=[12, 512, 64], help='Tensor Shape')
    args.add_argument('--dump_path', type=str, default='results')
    args = args.parse_args()
    size = args.size
    size_str = "x".join([str(i) for i in size])
    result_path = os.path.join(base_dir, args.dump_path, config_prefix, f"attention_{size_str}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    # setting environment variables
    os.environ['TORCHSIM_DUMP_PATH'] = result_path
    # only timing simulation
    os.environ['TORCHSIM_VALIDATION_MODE'] = "0"
    if 'BACKENDSIM_SPIKE_ONLY' in os.environ:
        del os.environ['BACKENDSIM_SPIKE_ONLY']

    run_attention(size, config)
