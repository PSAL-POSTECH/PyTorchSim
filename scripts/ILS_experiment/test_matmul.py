import torch
import argparse
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

def test_matmul(device, input_size=128, hidden_size=128, output_size=128):
    def custom_matmul(a, b):
        return torch.matmul(a, b)
    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(x1, w1)
    y = custom_matmul(x2, w2)
    test_result("Matmul Forward", res, y)

def test_addmm(device, input_size=128, hidden_size=128, output_size=128, bias_rank=1):
    def custom_matmul(bias, a, b):
        return torch.addmm(bias, a, b)
    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    bias = torch.randn(output_size) if bias_rank == 1 else torch.randn(input_size, output_size)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    b1 = bias.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    b2 = bias.to("cpu")
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(b1, x1, w1)
    y = custom_matmul(b2, x2, w2)
    test_result("Addmm Forward", res, y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run matmul with given shape") 
    parser.add_argument('--shape', type=str, default="(512,512,512)")
    args = parser.parse_args()
    shape = tuple(map(int, args.shape.strip('()').split(',')))
    device = torch.device("npu:0")
    test_matmul(device, *shape)