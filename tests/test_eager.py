import torch

if __name__ == "__main__":
    device = torch.device("npu:0")
    x = torch.zeros(10, 10).to(device)
    y = torch.zeros(10, 10).to(device)
    z = x + y
    print(z.cpu())