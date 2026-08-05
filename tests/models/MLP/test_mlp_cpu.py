import os
import shutil
import sys
import time
import contextlib
import unittest
import copy


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

class ExtendedMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(ExtendedMLP, self).__init__()

        # Check if hidden_sizes is valid
        if not hidden_sizes or len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain at least one layer size")

        self.layers = nn.ModuleList()

        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


def train_mlp_mnist(device):
    torch.manual_seed(0)
    num_samples = 128 * 10
    batch_size = 8

    epoch = 100
    iteration_per_epoch = num_samples // batch_size
    eval_size = 5

    """ Prepare Dataset """
    input_size = 28 * 28
    hidden_size = 32
    output_size = 10

    name = f"{batch_size}_{input_size}_{hidden_size}_{output_size}_pretrained_cpu"

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
    model_device = model.to(device=device)
    # model_device = torch.compile()(model_device)

    # save initial model state
    # torch.save(model_device.state_dict(), f"{name}/initial_model.pth")

    # Optional warm start; this run trains and saves its own best_model.pth.
    load_path = f"{name}/best_model.pth"
    if os.path.exists(load_path):
        model_device.load_state_dict(torch.load(load_path))
        print("model loaded..")

    model_device.train()

    criterion = nn.CrossEntropyLoss()
    # optimizer = Adam(model_device.parameters(), lr=0.001)
    optimizer = RMSprop(model_device.parameters(), lr=0.001)

    opt_zero_grad = torch.compile(optimizer.zero_grad, dynamic=False)

    loss_print_interval = 1
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
                # optimizer.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss.backward()
                # optimizer.step()

                if i % loss_print_interval == 0:
                    print(f"Train loss at epoch: {e}, iteration {i}: {loss.item()}")

                loss_list.append(loss.cpu().detach().numpy())
                with open(f"{name}/loss.txt", "a") as f:
                    f.write(str(loss.cpu().detach().numpy()) + "\n")

                # add train last loss list
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
                    print("Evaluation loss: ", loss.item())

            evaluation_loss = evaluation_loss / len(test_loader)
            evaluation_acc = evaluation_correct / evaluation_total
            print(f"Loss: {evaluation_loss}")
            print(f"Validation Accuracy: {evaluation_acc}")
            eval_loss_list.append(evaluation_loss)
            eval_acc_list.append(evaluation_acc)

            with open(f"{name}/eval_loss.txt", "a") as f:
                f.write(str(evaluation_loss) + "\n")
            with open(f"{name}/eval_acc.txt", "a") as f:
                f.write(str(evaluation_acc) + "\n")

            epochs_no_improve = 0
            # Early Stopping Logic
            if evaluation_loss < best_loss:
                best_loss = evaluation_loss
                no_improvement_epochs = 0
                # Save the best model if needed
                torch.save(model.state_dict(), f"{name}/best_model.pth")
                print("Improvement detected. Model saved.")
            else:
                no_improvement_epochs += 1
                print(f"No improvement for {no_improvement_epochs} epochs.")

            if no_improvement_epochs >= early_stop_patience:
                print("Early stopping triggered.")
                break



    train(model_device, device)

    return

# def train_mlp_cifar10(device):
#     torch.manual_seed(0)
#     num_samples = 128 * 30
#     batch_size = 32

#     epoch = 100
#     iteration_per_epoch = num_samples // batch_size
#     eval_size = 5

#     """ Prepare Dataset """
#     input_size = 3 * 32 * 32
#     # hidden_size = [1024, 1024, 512, 200]
#     hidden_size = [200, 200, 200, 200]
#     # hidden_size = [256, 128]
#     output_size = 10

#     name = f"cifar10_{batch_size}_{input_size}_{'_'.join(map(str, hidden_size))}_{output_size}_cpu"

#     # make dir with name
#     if not os.path.exists(name):
#         os.makedirs(name)

#     num_workers = 4
#     # Transforms for data augmentation and normalization
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),  # Random cropping with padding
#         transforms.RandomHorizontalFlip(),    # Random horizontal flipping
#         transforms.ToTensor(),                # Convert image to tensor
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalize with CIFAR-10 mean and std
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),                # Convert image to tensor
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalize with CIFAR-10 mean and std
#     ])

#     # Load CIFAR-100 dataset
#     train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
#     test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)

#     indices = [i for i, label in enumerate(train_dataset.targets)]
#     indices = indices[:num_samples]
#     subset_train_cifar = Subset(train_dataset, indices)

#     eval_indices = [i for i, label in enumerate(test_dataset.targets)]
#     eval_indices = eval_indices[:batch_size * eval_size]
#     subset_test_cifar = Subset(test_dataset, eval_indices)

#     # Create DataLoader
#     train_loader = DataLoader(subset_train_cifar, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     test_loader = DataLoader(subset_test_cifar, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     # Example: Iterate through the DataLoader
#     for images, labels in train_loader:
#         print(f"Image batch shape: {images.shape}")
#         print(f"Label batch shape: {labels.shape}")
#         break


#     """ Prepare Model """
#     model = ExtendedMLP(input_size, output_size, hidden_size)
#     model_device = model.to(device=device)
#     # model_device = torch.compile()(model_device)
#     summary(model_device, (input_size,))

#     # save initial model state
#     torch.save(model_device.state_dict(), f"{name}/initial_model.pth")

#     model_device.train()

#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model_device.parameters(), lr=0.001)
#     # optimizer = RMSprop(model_device.parameters(), lr=0.001)
#     opt_zero_grad = torch.compile()(optimizer.zero_grad)
#     opt_step = torch.compile()(optimizer.step)

#     loss_print_interval = 1
#     def train(model, device):
#         loss_list = []
#         train_last_loss_list = []
#         eval_loss_list = []
#         eval_acc_list = []

#         early_stop_patience = 5  # Number of epochs to wait before stopping
#         best_loss = float('inf')  # Initialize the best validation loss
#         best_acc = 0
#         no_improvement_epochs = 0  # Counter for epochs with no improvement

#         model.train()

#         for e in range(epoch):
#             for i, (x, y) in enumerate(train_loader):
#                 x = x.view(-1, input_size).to(device=device)
#                 y = y.to(device=device)
#                 opt_zero_grad()
#                 # optimizer.zero_grad()
#                 y_hat = model(x)
#                 loss = criterion(y_hat, y)
#                 loss.backward()
#                 opt_step()
#                 # optimizer.step()

#                 if i % loss_print_interval == 0:
#                     print(f"Train loss at epoch: {e}, iteration {i}: {loss.item()}")

#                 loss_list.append(loss.cpu().detach().numpy())
#                 with open(f"{name}/loss.txt", "a") as f:
#                     f.write(str(loss.cpu().detach().numpy()) + "\n")

#                 # add train last loss list
#                 if i % iteration_per_epoch == 0:
#                     train_last_loss_list.append(loss.cpu().detach().numpy())
#                     with open(f"{name}/train_last_loss.txt", "a") as f:
#                         f.write(str(loss.cpu().detach().numpy()) + "\n")


#             print(f"Evaluation at epoch {e}")
#             evaluation_loss = 0
#             evaluation_total = 0
#             evaluation_correct = 0
#             model.eval()
#             with torch.no_grad():
#                 for x, y in test_loader:
#                     x = x.view(-1, input_size).to(device=device)
#                     y = y.to(device=device)
#                     y_hat = model(x)
#                     loss = criterion(y_hat, y)
#                     evaluation_loss += loss.item()
#                     _, predicted = torch.max(y_hat.cpu().data, 1)
#                     evaluation_total += y.size(0)
#                     evaluation_correct += (predicted == y).sum().item()
#                     print("Evaluation loss: ", loss.item())

#             evaluation_loss = evaluation_loss / len(test_loader)
#             evaluation_acc = evaluation_correct / evaluation_total
#             print(f"Loss: {evaluation_loss}")
#             print(f"Validation Accuracy: {evaluation_acc}")
#             eval_loss_list.append(evaluation_loss)
#             eval_acc_list.append(evaluation_acc)

#             with open(f"{name}/eval_loss.txt", "a") as f:
#                 f.write(str(evaluation_loss) + "\n")
#             with open(f"{name}/eval_acc.txt", "a") as f:
#                 f.write(str(evaluation_acc) + "\n")

#             if evaluation_acc > best_acc:
#                 best_acc = evaluation_acc
#             # Early Stopping Logic
#             if evaluation_loss < best_loss:
#                 best_loss = evaluation_loss
#                 no_improvement_epochs = 0
#                 # Save the best model if needed
#                 torch.save(model.state_dict(), f"{name}/best_model.pth")
#                 print("Improvement detected. Model saved.")
#             else:
#                 no_improvement_epochs += 1
#                 print(f"No improvement for {no_improvement_epochs} epochs.")

#             if no_improvement_epochs >= early_stop_patience:
#                 print("Early stopping triggered.")
#                 break

#         print(f"Best validation loss: {best_loss}")
#         print(f"Best validation accuracy: {best_acc}")

#     train(model_device, device)

#     return


if __name__ == "__main__":
    # torch.set_printoptions(threshold=float('inf'), linewidth=600)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_mlp_mnist(device)
    # train_mlp_cifar10(device)
