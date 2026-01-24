import torch

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    message = f"|{name} Test Passed|"
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        print("Failed")
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        # exit(1)

def test_conv_residual(device, batch_size=1, in_channels=8, out_channels=16, input_size=64, kernel_size=3, stride=1, padding=0):
    def custom_conv2d(a, b, bias, c):
        i_c = a.shape[1]
        o_c = b.shape[0]
        conv2d = torch.nn.Conv2d(i_c, o_c, b.shape[-1], stride=stride, padding=padding, dilation=1, bias=True)
        conv2d.weight = torch.nn.Parameter(b)
        conv2d.bias = torch.nn.Parameter(bias)
        return conv2d(a) + c
    torch.manual_seed(0)
    conv_input = torch.randn(batch_size, in_channels, input_size, input_size).to(memory_format=torch.channels_last, device=device)
    conv_kernel = torch.randn(out_channels, in_channels, kernel_size, kernel_size).to(memory_format=torch.channels_last, device=device)
    conv_bias = torch.randn(out_channels).to(device=device)
    o_h = (input_size + 2 * padding - kernel_size) // stride + 1
    o_w = (input_size + 2 * padding - kernel_size) // stride + 1
    add_tensor = torch.randn(batch_size, out_channels, o_h, o_w).to(device=device)
    opt_fn = torch.compile(dynamic=False)(custom_conv2d)
    res = opt_fn(conv_input, conv_kernel, conv_bias, add_tensor)
    out = custom_conv2d(conv_input.cpu(), conv_kernel.cpu(), conv_bias.cpu(), add_tensor.cpu())
    test_result("Conv2d Residual Fusion Forward", res, out, rtol=1e-3, atol=1e-3)
    print("Max diff > ", torch.max(torch.abs(res.cpu() - out)))


def test_conv_scalar(device, batch_size=1, in_channels=8, out_channels=16, input_size=64, kernel_size=3, stride=1, padding=0):
    def custom_conv2d(a, b, bias, c):
        i_c = a.shape[1]
        o_c = b.shape[0]
        conv2d = torch.nn.Conv2d(i_c, o_c, b.shape[-1], stride=stride, padding=padding, dilation=1, bias=False)
        conv2d.weight = torch.nn.Parameter(b)
        # conv2d.bias = torch.nn.Parameter(bias)
        return conv2d(a) * c
    torch.manual_seed(0)
    conv_input = torch.randn(batch_size, in_channels, input_size, input_size).to(memory_format=torch.channels_last, device=device)
    conv_kernel = torch.randn(out_channels, in_channels, kernel_size, kernel_size).to(memory_format=torch.channels_last, device=device)
    conv_bias = torch.randn(out_channels).to(device=device)
    opt_fn = torch.compile(dynamic=False)(custom_conv2d)
    res = opt_fn(conv_input, conv_kernel, conv_bias, 2)
    out = custom_conv2d(conv_input.cpu(), conv_kernel.cpu(), conv_bias.cpu(), 2)
    test_result("Conv2d + Scalar Fusion Forward", res, out, rtol=1e-3, atol=1e-3)
    print("Max diff > ", torch.max(torch.abs(res.cpu() - out)))

def test_conv_relu(device, batch_size=1, in_channels=8, out_channels=16, input_size=64, kernel_size=3, stride=1, padding=0):
    def custom_conv2d(a, b, bias):
        i_c = a.shape[1]
        o_c = b.shape[0]
        conv2d = torch.nn.Conv2d(i_c, o_c, b.shape[-1], stride=stride, padding=padding, dilation=1, bias=True)
        conv2d.weight = torch.nn.Parameter(b)
        conv2d.bias = torch.nn.Parameter(bias)
        return torch.nn.functional.relu(conv2d(a))
    torch.manual_seed(0)
    conv_input = torch.randn(batch_size, in_channels, input_size, input_size).to(memory_format=torch.channels_last, device=device)
    conv_kernel = torch.randn(out_channels, in_channels, kernel_size, kernel_size).to(memory_format=torch.channels_last, device=device)
    conv_bias = torch.randn(out_channels).to(device=device)
    opt_fn = torch.compile(dynamic=False)(custom_conv2d)
    res = opt_fn(conv_input, conv_kernel, conv_bias)
    out = custom_conv2d(conv_input.cpu(), conv_kernel.cpu(), conv_bias.cpu())
    test_result("Conv2d + ReLU Fusion Forward", res, out, rtol=1e-3, atol=1e-3)
    print("Max diff > ", torch.max(torch.abs(res.cpu() - out)))

def test_conv_bn_relu(device, batch_size=1, in_channels=8, out_channels=16, input_size=64, kernel_size=3, stride=1, padding=0):
    def custom_conv_bn_relu(a, b, bias, c, d, e, f):
        i_c = a.shape[1]
        o_c = b.shape[0]
        conv2d = torch.nn.Conv2d(in_channels, out_channels, b.shape[-1], stride=stride, padding=padding, dilation=1, bias=True).eval()
        conv2d.weight = torch.nn.Parameter(b)
        conv2d.bias = torch.nn.Parameter(bias)
        # return torch.nn.functional.batch_norm(conv2d(a), c, d, weight=e, bias=f)
        return torch.nn.functional.relu(torch.nn.functional.batch_norm(conv2d(a), c, d, weight=e, bias=f))
    torch.manual_seed(0)
    conv_input = torch.randn(batch_size, in_channels, input_size, input_size).to(memory_format=torch.channels_last, device=device)
    conv_kernel = torch.randn(out_channels, in_channels, kernel_size, kernel_size).to(memory_format=torch.channels_last, device=device)
    conv_bias = torch.randn(out_channels).to(device=device)
    bn_weight = torch.randn(out_channels).to(device=device)
    bn_bias = torch.randn(out_channels).to(device=device)
    bn_mean = torch.zeros(out_channels).to(device=device)
    bn_var = torch.ones(out_channels).to(device=device)
    opt_fn = torch.compile(dynamic=False)(custom_conv_bn_relu)
    with torch.no_grad():
        res = opt_fn(conv_input, conv_kernel, conv_bias, bn_mean, bn_var, bn_weight, bn_bias)
    out = custom_conv_bn_relu(conv_input.cpu(), conv_kernel.cpu(), conv_bias.cpu(), bn_mean.cpu(), bn_var.cpu(), bn_weight.cpu(), bn_bias.cpu())
    test_result("Conv2d + BN + ReLU Fusion Forward", res, out, rtol=1e-3, atol=1e-3)
    print("Max diff > ", torch.max(torch.abs(res.cpu() - out)))

if __name__ == "__main__":
    device = torch.device("npu:0")

    # Vanila test
    test_conv_residual(device, batch_size=3, in_channels=64, out_channels=64, input_size=28, kernel_size=3, stride=1, padding=1)

    # Multi-tile test
    test_conv_residual(device, batch_size=1, in_channels=3, out_channels=32, input_size=32, kernel_size=3, stride=1, padding=1)

    # Single batch test
    test_conv_residual(device, batch_size=1, in_channels=16, out_channels=16, input_size=64, kernel_size=3, stride=1, padding=1)

    # Scalar
    test_conv_scalar(device, batch_size=1, in_channels=16, out_channels=48, input_size=48, kernel_size=3, stride=1, padding=1)

    # Relu
    test_conv_relu(device, batch_size=1, in_channels=16, out_channels=16, input_size=64, kernel_size=3, stride=1, padding=1)

    # Conv + BN + ReLU
    test_conv_bn_relu(device, batch_size=1, in_channels=8, out_channels=16, input_size=64, kernel_size=3, stride=1, padding=1)