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

class MLP(torch.nn.Module):
    def __init__(self, input_size=28*28, hidden_size=64, output_size=8):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # x = self.relu(x)
        # x = self.softmax(x)
        return x

def test_mlp(device, batch_size=64, input_size=64, hidden_size=32, output_size=8):
    torch.manual_seed(0)
    input = torch.randn(batch_size, input_size)
    x1 = copy.deepcopy(input).to(device=device)
    x2 = copy.deepcopy(input).to("cpu")
    target = torch.randn(batch_size, output_size)
    y1 = copy.deepcopy(target).to(device=device)
    y2 = copy.deepcopy(target).to("cpu")
    model = MLP(input_size, hidden_size, output_size)
    model.requires_grad = True
    model.to(device=device)
    opt_fn = torch.compile(dynamic=False)(model)
    y = opt_fn(x1)
    cpu_model = copy.deepcopy(model).to("cpu")
    cpu_model.requires_grad = True
    cpu_y = cpu_model(x2)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt_loss = torch.compile(dynamic=False)(loss_fn)
    loss = loss_fn(y, y1)
    cpu_loss = loss_fn(cpu_y, y2)
    loss.backward()
    cpu_loss.backward()
    test_result("MLP Forward", y, cpu_y)
    test_result("Loss", loss, cpu_loss)
    test_result("MLP Weight1 Backward", model.linear1.weight.grad, cpu_model.linear1.weight.grad)
    test_result("MLP Bias1 Backward", model.linear1.bias.grad, cpu_model.linear1.bias.grad)
    test_result("MLP Weight2 Backward", model.linear2.weight.grad, cpu_model.linear2.weight.grad)
    test_result("MLP Bias2 Backward", model.linear2.bias.grad, cpu_model.linear2.bias.grad)

def test_mlp_inf(device, batch_size=64, input_size=64, hidden_size=32, output_size=8, sparsity=0.0):
    torch.manual_seed(0)
    input = torch.randn(batch_size, input_size)
    x1 = copy.deepcopy(input).to(device=device)
    x2 = copy.deepcopy(input).to("cpu")
    target = torch.randn(batch_size, output_size)
    model = MLP(input_size, hidden_size, output_size)
    model.requires_grad = False
    model.to(device=device)
    opt_fn = torch.compile(dynamic=False)(model)
    y = opt_fn(x1)
    cpu_model = copy.deepcopy(model).to("cpu")
    cpu_model.requires_grad = False
    cpu_y = cpu_model(x2)
    test_result("MLP Forward", y, cpu_y)

def test_optimizer(device):
    torch.manual_seed(0)
    model = MLP(input_size=16, hidden_size=16, output_size=16).to(device=device)
    model.requires_grad = True
    cpu_model = copy.deepcopy(model).to("cpu")
    opt_model = torch.compile(dynamic=False)(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cpu_optimizer = torch.optim.Adam(cpu_model.parameters(), lr=0.001)
    opt_step = torch.compile(dynamic=False)(optimizer.step)
    input = torch.randn(16, 16)
    x1 = copy.deepcopy(input).to(device=device)
    x2 = copy.deepcopy(input).to("cpu")
    y = opt_model(x1)
    cpu_y = cpu_model(x2)
    loss = y.sum()
    cpu_loss = cpu_y.sum()
    optimizer.zero_grad()
    cpu_optimizer.zero_grad()
    loss.backward()
    cpu_loss.backward()
    opt_step()
    cpu_optimizer.step()
    test_result("Optimizer", model.linear1.weight, cpu_model.linear1.weight)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_mlp(device)
    test_mlp_inf(device, batch_size=1, input_size=256, hidden_size=512, output_size=256)
    test_mlp_inf(device, batch_size=8, input_size=256, hidden_size=512, output_size=256)
    test_mlp_inf(device, batch_size=64, input_size=256, hidden_size=512, output_size=256)
    test_optimizer(device)
