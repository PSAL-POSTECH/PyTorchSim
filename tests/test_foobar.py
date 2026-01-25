import torch
import torch._dynamo

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

def test_foobar(device, size=(128, 128)):
    def vector_foobar(a):
        return torch._foobar(a)

    x = torch.randn(size).to(device=device)
    opt_fn = torch.compile(dynamic=False)(vector_foobar)
    res = opt_fn(x)

    out = x.cpu()
    test_result("Foobar", res, out)


if __name__ == "__main__":
    import os
    import sys
    import argparse
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    parser = argparse.ArgumentParser(description="Run Foobar test with dynamic shape")
    parser.add_argument('--shape', type=str, default="(512,768)")
    args = parser.parse_args()
    shape = tuple(map(int, args.shape.strip('()').split(',')))

    from Scheduler.scheduler import PyTorchSimRunner
    module = PyTorchSimRunner.setup_device()
    device = module.custom_device()
    test_foobar(device, (1, 1))
    test_foobar(device, (47, 10))
    test_foobar(device, (128, 128))
    test_foobar(device, shape)