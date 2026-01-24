import copy
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

class Matmul_ActivationFn(torch.nn.Module):
    def __init__(self, input_size, output_size, activation_fn):
        super(Matmul_ActivationFn, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)
        if activation_fn == "relu":
            self.activation_fn = torch.nn.ReLU()
        elif activation_fn == "sigmoid":
            self.activation_fn = torch.nn.Sigmoid()
        else:
            NotImplementedError("Activation function not implemented")

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        return x

class Matmul_Residual_ActivationFn(torch.nn.Module):
    def __init__(self, input_size, output_size, activation_fn):
        super(Matmul_Residual_ActivationFn, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)
        if activation_fn == "relu":
            self.activation_fn = torch.nn.ReLU()
        elif activation_fn == "sigmoid":
            self.activation_fn = torch.nn.Sigmoid()
        else:
            NotImplementedError("Activation function not implemented")

    def forward(self, x, residual):
        x = self.linear1(x) + residual
        x = self.activation_fn(x)
        return x

def test_matmul_activation(device, batch_size=16, input_size=32, output_size=8, activation_fn="relu"):
    torch.manual_seed(0)
    input = torch.randn(batch_size, input_size)
    if device:
        x1 = copy.deepcopy(input).to(device=device)
    x2 = copy.deepcopy(input).to("cpu")
    model = Matmul_ActivationFn(input_size, output_size, activation_fn)
    if device:
        model.to(device=device)
        opt_fn = torch.compile(dynamic=False)(model)
        y = opt_fn(x1)
    cpu_model = copy.deepcopy(model).to("cpu")
    cpu_y = cpu_model(x2)
    if device:
        test_result(f"Matmul_ActivationFn {activation_fn}", y, cpu_y)
    else:
        print("CPU output > ", cpu_y)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_matmul_activation(device)
    test_matmul_activation(device, batch_size=32, input_size=32, output_size=32, activation_fn="sigmoid")
    test_matmul_activation(device, batch_size=42, input_size=42, output_size=42, activation_fn="sigmoid")
