import os
import shutil
import sys
import time
import contextlib
import unittest
import copy
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.optim import Adam, RMSprop
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch._dynamo
import torch.utils.cpp_extension
from torch._inductor import config

sys.path.insert(0, os.path.join(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'), 'tests'))
from _pytorchsim_utils import test_result

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out

def test_mlp(device):
    torch.manual_seed(0)
    batch_size = 1256
    input_size = 128
    hidden_size = 128
    output_size = 128

    input = torch.randn(batch_size, input_size)
    x1 = copy.deepcopy(input).to(device=device)
    x2 = copy.deepcopy(input).to("cpu")

    model = MLP(input_size, output_size, hidden_size)
    model_cpu = copy.deepcopy(model).to("cpu")

    model_device = model.to(device=device)
    model_device = torch.compile(model_device, dynamic=False)

    y1 = model_device(x1)
    y2 = model_cpu(x2)

    test_result("MLP", y1, y2)

def test_train_mlp(device):
    torch.manual_seed(0)
    batch_size = 200
    input_size = 28 * 28
    hidden_size = 32
    output_size = 10

    input = torch.randn(batch_size, input_size)
    target = torch.randn(batch_size, output_size)
    x1 = copy.deepcopy(input).to(device=device)
    x2 = copy.deepcopy(input).to("cpu")
    y1 = copy.deepcopy(target).to(device=device)
    y2 = copy.deepcopy(target).to("cpu")

    model = MLP(input_size, output_size, hidden_size)
    model_cpu = copy.deepcopy(model).to("cpu")

    model_device = model.to(device=device)
    model_device = torch.compile(model_device, dynamic=False)

    model_device.eval()
    model_cpu.eval()
    # model_device.train()
    # model_cpu.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = RMSprop(model_device.parameters(), lr=0.001)
    opt_zero_grad = torch.compile(optimizer.zero_grad, dynamic=False)
    opt_step = torch.compile(optimizer.step, dynamic=False)

    """ Forward """
    print("Forward")
    y1_hat = model_device(x1)
    y2_hat = model_cpu(x2)
    test_result("MLP Forward", y1_hat, y2_hat)

    """ Loss """
    print("Loss")
    loss1 = criterion(y1_hat, y1)
    loss2 = criterion(y2_hat, y2)
    test_result("Loss", loss1, loss2)

    """ Backward """
    opt_zero_grad()

    print("Backward")
    loss1.backward()
    model_cpu.zero_grad()
    loss2.backward()

    """ Optimize """
    print("Optimize")
    opt_step()

    # Check weights and gradients
    # for p1, p2 in zip(model_device.parameters(), model_cpu.parameters()):
    #     test_result("Gradient", p1.grad, p2.grad)
    # test_result("FC1 Gradient", model_device.fc1.weight.grad, model_cpu.fc1.weight.grad)
    # test_result("FC1 Bias Gradient", model_device.fc1.bias.grad, model_cpu.fc1.bias.grad)
    # test_result("FC2 Gradient", model_device.fc2.weight.grad, model_cpu.fc2.weight.grad)
    # test_result("FC2 Bias Gradient", model_device.fc2.bias.grad, model_cpu.fc2.bias.grad)
    print("Finished")

def train_mlp_mnist(device):
    torch.manual_seed(0)
    num_samples = 128 * 10
    batch_size = 256

    epoch = 100
    iteration_per_epoch = num_samples // batch_size
    eval_size = 5

    """ Prepare Dataset """
    input_size = 28 * 28
    hidden_size = 32
    output_size = 10

    name = f"{batch_size}_{input_size}_{hidden_size}_{output_size}"
    # make dir with name
    if not os.path.exists(name):
        os.makedirs(name)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
    train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)

    indices = [i for i, label in enumerate(train_dataset.targets)]
    indices = indices[:num_samples]
    subset_train_mnist = Subset(train_dataset, indices)

    eval_indices = [i for i, label in enumerate(test_dataset.targets)]
    eval_indices = eval_indices[:batch_size * eval_size]
    subset_test_mnist = Subset(test_dataset, eval_indices)

    train_loader = DataLoader(dataset=subset_train_mnist, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=subset_test_mnist, batch_size=batch_size, shuffle=False)


    """ Prepare Model """
    model = MLP(input_size, output_size, hidden_size)

    # load inital model state from path
    inital_model_path = f"./{name}_cpu/initial_model.pth"
    model.load_state_dict(torch.load(inital_model_path))
    print("Model load complete.")

    model_device = model.to(device=device)
    model_device = torch.compile(model_device, dynamic=False)

    model_device.train()

    criterion = nn.CrossEntropyLoss()
    # optimizer = Adam(model_device.parameters(), lr=0.001)
    optimizer = RMSprop(model_device.parameters(), lr=0.001)
    opt_zero_grad = torch.compile(optimizer.zero_grad, dynamic=False)
    opt_step = torch.compile(optimizer.step, dynamic=False)

    loss_print_interval = 1
    evaluation_interval = 5
    saturate_epochs = 3
    def train(model, device):
        loss_list = []
        train_last_loss_list = []
        eval_loss_list = []
        eval_acc_list = []

        early_stop_patience = 5  # Number of epochs to wait before stopping
        best_loss = float('inf')  # Initialize the best validation loss
        no_improvement_epochs = 0  # Counter for epochs with no improvement

        model.train()

        for e in range(epoch):
            for i, (x, y) in enumerate(train_loader):
                x = x.view(-1, input_size).to(device=device)
                y = y.to(device=device)
                opt_zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss.backward()
                opt_step()

                if i % loss_print_interval == 0:
                    print(f"Train loss at epoch: {e}, iteration {i}: {loss.item()}")

                loss_list.append(loss.cpu().detach().numpy())
                with open(f"{name}/loss.txt", "a") as f:
                    f.write(str(loss.cpu().detach().numpy()) + "\n")

                if i % iteration_per_epoch == 0:
                    train_last_loss_list.append(loss.cpu().detach().numpy())
                    with open(f"{name}/train_last_loss.txt", "a") as f:
                        f.write(str(loss.cpu().detach().numpy()) + "\n")


            print(f"Evaluation at epoch {e}")
            evaluation_loss = 0
            evaluation_total = 0
            evaluation_correct = 0
            model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.view(-1, input_size).to(device=device)
                    y = y.to(device=device)
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                    evaluation_loss += loss.item()
                    _, predicted = torch.max(y_hat.cpu().data, 1)
                    evaluation_total += y.size(0)
                    evaluation_correct += (predicted == y).sum().item()

            evaluation_loss = evaluation_loss / len(test_loader)
            evaluation_acc = evaluation_correct / evaluation_total

            print(f"Train loss: {evaluation_loss}")
            print(f"Validation Accuracy: {evaluation_acc}")
            eval_loss_list.append(evaluation_loss)
            eval_acc_list.append(evaluation_acc)

            with open(f"{name}/eval_loss.txt", "a") as f:
                f.write(str(evaluation_loss / len(test_loader)) + "\n")
            with open(f"{name}/eval_acc.txt", "a") as f:
                f.write(str(evaluation_correct / evaluation_total) + "\n")

            epochs_no_improve = 0
            # Early Stopping Logic
            if evaluation_loss < best_loss:
                best_loss = evaluation_loss
                no_improvement_epochs = 0
                # Save the best model if needed
                torch.save({key: value.cpu() for key, value in model.state_dict().items()}, f"{name}/best_model.pth")
                print("Improvement detected. Model saved.")
            else:
                no_improvement_epochs += 1
                print(f"No improvement for {no_improvement_epochs} epochs.")

            if no_improvement_epochs >= early_stop_patience:
                print("Early stopping triggered.")
                break

    train(model_device, device)

    return

if __name__ == "__main__":
    torch.set_printoptions(threshold=float('inf'), linewidth=600)
    device = torch.device("npu:0")

    test_mlp(device)
    # test_train_mlp(device)
    # train_mlp_mnist(device)
