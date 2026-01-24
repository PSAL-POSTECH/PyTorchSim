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

def test_maxpool(device, b=1, c=64, h=112, w=112):
    torch.manual_seed(0)
    model = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1).eval()
    model.to(device=device)
    input = torch.randn(b, c, h, w).to(device=device)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    opt_fn = torch.compile(dynamic=False)(model)
    res = opt_fn(x1)
    model.to("cpu")
    out = model(x2)
    test_result("Maxpool Forward", res, out) # TODO: MaxPool Functionality is not working

def test_avgpool(device, b=1, c=64, h=112, w=112):
    def avgpool(a):
        return torch.nn.AdaptiveAvgPool2d((1, 1))(a)
    torch.manual_seed(0)
    input = torch.randn(b, c, h, w).to(device=device) #FIXME: channel 8 does not work (range padding issue)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    opt_fn = torch.compile(dynamic=False)(avgpool)
    res = opt_fn(x1)
    out = avgpool(x2)
    test_result("Avgpool Forward", res, out)

if __name__ == "__main__":
    device = torch.device("npu:0")
    #test_maxpool(device, b=1, c=8, h=16, w=16)
    #test_maxpool(device, b=1, c=8, h=112, w=112)
    test_avgpool(device, b=1, c=512, h=7, w=7)
