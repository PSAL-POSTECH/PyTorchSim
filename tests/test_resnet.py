import argparse
import torch
import torch._dynamo
import torch.utils.cpp_extension
from torchvision.models import resnet18, resnet50
import os
import sys
sys.path.insert(0, os.path.join(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'), 'tests'))
from _pytorchsim_utils import test_result

def test_resnet(device, batch=1, model_type='resnet18'):
    from torchvision.models import resnet
    with torch.no_grad():
        #model = resnet._resnet(resnet.BasicBlock, [1, 1, 1, 1], weights=None, progress=False).eval()
        if model_type == 'resnet50':
            model = resnet50().eval()
        elif model_type == 'resnet18':
            model = resnet18().eval()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        model.to(device, memory_format=torch.channels_last)
        input = torch.randn(batch, 3, 224, 224)
        x1 = input.to(device=device, memory_format=torch.channels_last)
        x2 = input.cpu().to(memory_format=torch.channels_last)
        opt_fn = torch.compile(dynamic=False)(model)
        res = opt_fn(x1)
        cpu_model = model.cpu().to(memory_format=torch.channels_last)
        cpu_res = cpu_model(x2)
    test_result(f"{model_type} inference", res, cpu_res)
    print("Max diff > ", torch.max(torch.abs(res.cpu() - cpu_res)))
    print(f"{model_type} Simulation Done")

if __name__ == "__main__":
    import os
    import sys
    args = argparse.ArgumentParser()
    args.add_argument('--model_type', type=str, default="resnet18", help='ex) resnet18')
    args = args.parse_args()

    device = torch.device("npu:0")
    test_resnet(device, model_type=args.model_type)
