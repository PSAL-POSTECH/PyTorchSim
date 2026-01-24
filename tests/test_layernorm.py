import torch
import torch._dynamo
import torch.utils.cpp_extension

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

def test_LayerNorm(device, size=(64, 64)):
    torch.manual_seed(0)
    input = torch.randn(size)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    model = torch.nn.LayerNorm(size[-1])
    model.to(device=device)
    opt_fn = torch.compile(dynamic=False)(model)
    y = opt_fn(x1)
    cpu_model = model.to("cpu")
    cpu_y = cpu_model(x2)
    test_result("LayerNorm Forward", y, cpu_y)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LayerNorm test with dynamic shape")
    parser.add_argument('--shape', type=str, help="Shape of the tensor in the format (batch_size, features)", default="(512,768)")
    args = parser.parse_args()
    shape = tuple(map(int, args.shape.strip('()').split(',')))

    device = torch.device("npu:0")
    with torch.no_grad():
        #test_LayerNorm(device)
        test_LayerNorm(device, shape)
