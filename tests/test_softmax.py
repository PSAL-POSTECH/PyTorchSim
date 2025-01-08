import torch
import torch._dynamo
import torch.utils.cpp_extension

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    message = f"|{name} Test Passed|"
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

def test_softmax(device, size=(128, 128), dim=1):
    torch.manual_seed(0)
    input = torch.randn(size)
    x1 = input.to(device=device)
    x2 = input.to("cpu")

    # split softmax into 3 steps
    # def softmax1(x): # find max
    #     return x.max(dim=dim, keepdim=True).values
    # def softmax2(x, max):
    #     return (x - max).exp().sum(dim=dim, keepdim=True)
    # def softmax3(x, max, sum):
    #     return (x - max).exp().div(sum)

    # opt_fn1 = torch.compile(dynamic=False)(softmax1)
    # opt_fn2 = torch.compile(dynamic=False)(softmax2)
    # opt_fn3 = torch.compile(dynamic=False)(softmax3)

    # max = opt_fn1(x1)
    # cpu_max = softmax1(x2)
    # test_result("Softmax Max", max, cpu_max)
    # sum = opt_fn2(x1, max)
    # cpu_sum = softmax2(x2, cpu_max)
    # test_result("Softmax Sum", sum, cpu_sum)
    # y = opt_fn3(x1, max, sum)
    # cpu_y = softmax3(x2, cpu_max, cpu_sum)
    # test_result("Softmax", y, cpu_y)

    opt_fn = torch.compile(dynamic=False)(torch.nn.functional.softmax)
    y = opt_fn(x1, dim=dim)
    cpu_y = torch.nn.functional.softmax(x2, dim=dim)
    test_result("Softmax", y, cpu_y)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_softmax(device, size=(64, 128))
    test_softmax(device, size=(256, 128))
    test_softmax(device, size=(1, 16))
