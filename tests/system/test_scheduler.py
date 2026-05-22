import os
import sys
import torch
from torchvision.models import resnet18 as model1

base_path = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.append(base_path)
from models.test_transformer import EncoderBlock as model2
from Simulator.simulator import TOGSimulator

config = f'{base_path}/configs/systolic_ws_128x128_c2_simple_noc_tpuv3_partition.yml'

target_model1 = model1().eval()
target_model2 = model2(768, 12).eval()

device = torch.device("npu:0")
opt_model1 = torch.compile(target_model1.to(device=device, memory_format=torch.channels_last))
opt_model2 = torch.compile(target_model2.to(device=device))
model_input1 = torch.randn(1, 3, 224, 224).to(device=device)
model_input2 = torch.randn(128, 768).to(device=device)

with TOGSimulator(config_path=config):
    torch.npu.launch_model(opt_model1, model_input1, stream_index=0, timestamp=0)
    torch.npu.launch_model(opt_model2, model_input2, stream_index=1, timestamp=0)
    torch.npu.synchronize()
    torch.npu.launch_model(opt_model1, model_input1, stream_index=0, timestamp=0)
    torch.npu.launch_model(opt_model2, model_input2, stream_index=1, timestamp=0)
print("Done")
