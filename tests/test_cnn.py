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

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(8, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.norm = torch.nn.BatchNorm2d(16)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        return x

def test_CNN(device):
    torch.manual_seed(0)
    input = torch.randn(1, 8, 64, 64)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    model = CNN().eval()
    model.to(device=device)
    opt_fn = torch.compile(dynamic=False)(model)
    y = opt_fn(x1)
    cpu_model = model.to("cpu")
    cpu_y = cpu_model(x2)
    test_result("CNN Forward", y, cpu_y, rtol=2e-1, atol=2e-1)
    print("Max diff > ", torch.max(torch.abs(y.cpu() - cpu_y)))

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_CNN(device)
