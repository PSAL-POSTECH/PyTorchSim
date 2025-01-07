import copy
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

def test_single_perceptron(device):
    def perceptron(a, b, c):
        res = a * b + c
        return res

    def weight_update(a, b, lr):
        return a - b * lr
    from sklearn.datasets import make_regression
    X, Y = make_regression(n_samples=128, n_features=1, noise=30, random_state=1)
    input = torch.tensor(X.squeeze(-1), dtype=torch.float32)
    weight = torch.randn(1)
    x1 = copy.deepcopy(input).to(device=device)
    x2 = copy.deepcopy(input).to("cpu")
    w1 = copy.deepcopy(weight).to(device=device)
    w2 = copy.deepcopy(weight).to("cpu")
    target_y = torch.tensor(Y, dtype=torch.float32)
    y1 = copy.deepcopy(target_y).to(device=device)
    y2 = copy.deepcopy(target_y).to("cpu")
    b = torch.randn(1)
    b1 = copy.deepcopy(b).to(device=device)
    b2 = copy.deepcopy(b).to("cpu")
    w1.requires_grad = True
    w2.requires_grad = True
    b1.requires_grad = True
    b2.requires_grad = True
    opt_mlp = torch.compile(dynamic=False)(perceptron)
    opt_w = torch.compile(dynamic=False)(weight_update)
    loss_fn = torch.nn.MSELoss()
    opt_loss = torch.compile(dynamic=False)(loss_fn)
    lr = torch.tensor(5e-2).to(device=device) # learning rate
    y = opt_mlp(w1, x1, b1)
    loss = opt_loss(y, y1)
    loss.backward()
    cpu_y = perceptron(x2, w2, b2)
    cpu_loss = loss_fn(cpu_y, y2)
    cpu_loss.backward()
    test_result("Perceptron", y, cpu_y)
    test_result("Loss", loss, cpu_loss)
    test_result("Weight Update", w1.grad, w2.grad)
    test_result("Bias Update", b1.grad, b2.grad)
    # for i in range(50):
    #     y = opt_mlp(w1, x1, b1)
    #     loss = opt_loss(y, y1)
    #     # print(loss.cpu().item()) # check loss
    #     loss.to(device=device)
    #     loss.backward()
    #     with torch.no_grad():
    #         w1.copy_(opt_w(w1, w1.grad, lr))
    #         b1.copy_(opt_w(b1, b1.grad, lr))
    #     w1.grad.zero_()
    #     b1.grad.zero_()
    # # plot input and output on 2D plane, and plot the y = w*x + b line
    # plt.scatter(x1.cpu().numpy(), y1.cpu().numpy(), c='#c80151')
    # x = np.linspace(-3, 3, 100)
    # y = w1.cpu().item() * x + b1.cpu().item()
    # plt.plot(x, y, '-k')
    # plt.show()
    # plt.savefig('result.png')

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_single_perceptron(device)
