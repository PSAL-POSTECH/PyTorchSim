import torch
import torch._dynamo
import torch.utils.cpp_extension

import argparse
import datetime

def run_BERT(size, input_seq, config):
    from Scheduler.scheduler import Scheduler, SchedulerDNNModel, Request
    # from tests.test_transformer import DecoderBlock
    from tests.Fusion.test_transformer_fusion import DecoderBlock
    scheduler = Scheduler(num_request_queue=1, engine_select=Scheduler.FIFO_ENGINE, backend_config=config)
    device = scheduler.execution_engine.module.custom_device()

    hidden_dim = {'base': 768, 'large': 1024, 'xlarge': 2048}
    embedding_size = {'base': 768, 'large': 1024, 'xlarge': 2048}
    heads = {'base': 12, 'large': 16, 'xlarge': 32} # hidden/64 https://arxiv.org/pdf/1909.11942
    cpu_query = torch.randn(input_seq, hidden_dim[size])
    decoder_block = DecoderBlock(embedding_size[size], heads[size]).eval()

    query = cpu_query.clone().to(device=device)
    opt_fn = torch.compile(dynamic=False)(decoder_block.to(device=device))

    SchedulerDNNModel.register_model(f"BERT-{size}", opt_fn)
    request = Request(f"BERT-{size}", [query], [], request_queue_idx=0)
    scheduler.add_request(request, request_time=0)

    # Run scheduler
    while not scheduler.is_finished():
        with torch.no_grad():
            scheduler.schedule()

    print(f"BERT-{size} Simulation Done")

if __name__ == "__main__":
    import os
    import sys
    base_dir = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
    config = os.environ.get('TORCHSIM_CONFIG', default=f'{base_dir}/PyTorchSimBackend/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.json')
    config_prefix = config.split('/')[-1].split('.')[0][9:] # extract config name from config path FIXME: gem5 result is different as directoy name
    sys.path.append(base_dir)
    args = argparse.ArgumentParser()
    args.add_argument('--size', type=str, default='base')
    args.add_argument('--dump_path', type=str, default='results')
    args.add_argument('--input_size', type=int, default=512)
    args = args.parse_args()
    size = args.size
    input_seq = args.input_size
    result_path = os.path.join(base_dir, args.dump_path, config_prefix, f"BERT_{size}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    # setting environment variables
    os.environ['TORCHSIM_DUMP_PATH'] = result_path
    # only timing simulation
    os.environ['TORCHSIM_VALIDATION_MODE'] = "0"
    if 'BACKENDSIM_SPIKE_ONLY' in os.environ:
        del os.environ['BACKENDSIM_SPIKE_ONLY']

    run_BERT(size, input_seq, config)
