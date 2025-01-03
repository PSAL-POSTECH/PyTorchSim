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

def test_conv2d(device, batch_size=1, in_channels=8, out_channels=16, input_size=64, kernel_size=3, stride=1, padding=0):
    def custom_conv2d(a, b, bias):
        i_c = a.shape[1]
        o_c = b.shape[0]
        conv2d = torch.nn.Conv2d(i_c, o_c, b.shape[-1], stride=stride, padding=padding, dilation=1)
        conv2d.weight = torch.nn.Parameter(b)
        conv2d.bias = torch.nn.Parameter(bias)
        return conv2d(a)
    torch.manual_seed(0)
    conv_input = torch.randn(batch_size, in_channels, input_size, input_size).to(memory_format=torch.channels_last, device=device)
    conv_kernel = torch.randn(out_channels, in_channels, kernel_size, kernel_size).to(memory_format=torch.channels_last, device=device)
    conv_bias = torch.randn(out_channels).to(device=device)
    opt_fn = torch.compile(dynamic=False)(custom_conv2d)
    res = opt_fn(conv_input, conv_kernel, conv_bias)
    out = custom_conv2d(conv_input.cpu(), conv_kernel.cpu(), conv_bias.cpu())
    test_result("Conv2d Forward", res, out, rtol=1e-3, atol=1e-3)
    print("Max diff > ", torch.max(torch.abs(res.cpu() - out)))

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_conv2d(device, batch_size=1, in_channels=128, out_channels=128, input_size=28, kernel_size=3, stride=1, padding=1)
