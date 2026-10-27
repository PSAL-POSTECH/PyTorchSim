import argparse
import torch
import torch._dynamo
import torch.utils.cpp_extension
from torchvision.models import resnet18, resnet50
import os
import sys
sys.path.insert(0, os.path.join(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'), 'tests'))
from result_check import check_result, f32_reference

# (rtol, atol) per dtype, float32 at check_result's default; the bf16 atol is 3x the worst bf16 CPU diff.
_TOL = {"bfloat16": (5e-2, 0.5), "float32": (1e-4, 1e-4)}
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def test_resnet(device, batch=1, model_type='resnet18', dtype="bfloat16"):
    """torchvision `model_type` at 224 px, channels-last, with weights and input in `dtype`, against its
    float32 reference on the CPU."""
    from torchvision.models import resnet
    with torch.no_grad():
        #model = resnet._resnet(resnet.BasicBlock, [1, 1, 1, 1], weights=None, progress=False).eval()
        if model_type == 'resnet50':
            model = resnet50().eval()
        elif model_type == 'resnet18':
            model = resnet18().eval()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        model.to(_DTYPES[dtype])
        cpu_model = f32_reference(model).to(memory_format=torch.channels_last)
        model.to(device, memory_format=torch.channels_last)
        input = torch.randn(batch, 3, 224, 224).to(_DTYPES[dtype])
        x1 = input.to(device=device, memory_format=torch.channels_last)
        x2 = input.cpu().float().to(memory_format=torch.channels_last)
        opt_fn = torch.compile(dynamic=False)(model)
        res = opt_fn(x1)
        cpu_res = cpu_model(x2)
    check_result(f"{model_type} inference", res, cpu_res, *_TOL[dtype])
    print("Max diff > ", torch.max(torch.abs(res.cpu() - cpu_res)))
    print(f"{model_type} Simulation Done")

if __name__ == "__main__":
    import os
    import sys
    args = argparse.ArgumentParser()
    args.add_argument('--model_type', type=str, default="resnet18", help='ex) resnet18')
    args.add_argument("--dtype", type=str, default="bfloat16", choices=sorted(_DTYPES))
    args = args.parse_args()

    device = torch.device("npu:0")
    test_resnet(device, model_type=args.model_type, dtype=args.dtype)
