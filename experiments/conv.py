import torch
import torch._dynamo
import torch.utils.cpp_extension

import argparse
import datetime


def run_conv2d(batch_size, i_h, i_w, i_c, o_c, kernel_size, stride, padding, config):
    from Scheduler.scheduler import Scheduler, SchedulerDNNModel, Request
    def custom_conv2d(a, b, bias):
        i_c = a.shape[1]
        o_c = b.shape[0]
        conv2d = torch.nn.Conv2d(i_c, o_c, b.shape[-1], stride=stride, padding=padding, dilation=1, bias=False)
        conv2d.weight = torch.nn.Parameter(b)
        # conv2d.bias = torch.nn.Parameter(bias)
        return conv2d(a)
    scheduler = Scheduler(num_request_queue=1, engine_select=Scheduler.FIFO_ENGINE, togsim_config=config)
    device = scheduler.execution_engine.module.custom_device()
    conv_input = torch.randn(batch_size, i_c, i_h, i_w).to(memory_format=torch.channels_last, device=device)
    conv_kernel = torch.randn(o_c, i_c, kernel_size, kernel_size).to(memory_format=torch.channels_last, device=device)
    conv_bias = torch.randn(o_c).to(device=device)
    opt_fn = torch.compile(dynamic=False)(custom_conv2d)

    SchedulerDNNModel.register_model("CONV", opt_fn)
    request = Request("CONV", [conv_input, conv_kernel, conv_bias], [], request_queue_idx=0)
    scheduler.add_request(request, request_time=0)

    # Run scheduler
    while not scheduler.is_finished():
        with torch.no_grad():
            scheduler.schedule()

    print(f"CONV {batch_size}_{i_h}_{i_w}_{i_c}_{o_c}_{kernel_size}_{stride}_{padding} (B_H_W_I_C_O_C_K_S_P) Simulation Done")

if __name__ == "__main__":
    import os
    import sys
    base_dir = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
    config = os.environ.get('TORCHSIM_CONFIG', default=f'{base_dir}/configs/systolic_ws_128x128_c2_simple_noc_tpuv4.yml')
    config_prefix = config.split('/')[-1].split('.')[0][9:] # extract config name from config path
    sys.path.append(base_dir)
    args = argparse.ArgumentParser()
    args.add_argument('--size', nargs='+', type=int, default=[8, 28, 28, 128, 128, 3, 1, 1], help='B H W I_C O_C K S P')
    args.add_argument('--dump_path', type=str, default='results')
    args = args.parse_args()
    size = args.size
    size_str = "_".join([str(i) for i in size])
    result_path = os.path.join(base_dir, args.dump_path, config_prefix, f"CONV_{size_str}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    # setting environment variables
    os.environ['TORCHSIM_LOG_PATH'] = result_path
    # only timing simulation
    os.environ['TORCHSIM_VALIDATION_MODE'] = "0"
    if 'pytorchsim_functional_mode' in os.environ:
        del os.environ['pytorchsim_functional_mode']

    run_conv2d(size[0], size[1], size[2], size[3], size[4], size[5], size[6], size[7], config)