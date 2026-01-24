import torch

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

def test_BatchNorm(device, size=(1, 16, 64, 64)):
    torch.manual_seed(0)
    model = torch.nn.BatchNorm2d(size[1]).eval()
    model.to(device=device, memory_format=torch.channels_last)
    input = torch.empty_strided(size, (size[1]*size[2]*size[3], 1, size[1], size[1]*size[2]))
    input.uniform_(-1, 1)

    x1 = input.to(device=device, memory_format=torch.channels_last)
    x2 = input.to("cpu", memory_format=torch.channels_last)
    opt_fn = torch.compile(dynamic=False)(model)
    y = opt_fn(x1)
    cpu_model = model.to("cpu")
    cpu_y = cpu_model(x2)
    test_result("BatchNorm Forward", y, cpu_y)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_BatchNorm(device)
    test_BatchNorm(device, size=(1,64, 32, 32))
    test_BatchNorm(device, size=(1, 8, 4, 4))
    test_BatchNorm(device, size=(1,256, 32, 32))
