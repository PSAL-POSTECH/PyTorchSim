import torch

torch.npu.register_eager_to_compile(["aten::mul.Tensor", "aten::add.Tensor"])

if __name__ == "__main__":
    #torch.npu.register_fallback_op("aten::add.out", my_fallback)
    device = torch.device("npu:0")
    x = torch.ones(10, 10).to(device)
    y = torch.ones(10, 10).to(device)
    z = x * y
    z = x + z
    print(z.cpu())