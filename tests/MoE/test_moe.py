# Owner(s): ["module: inductor"]
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
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch._dynamo
import torch.utils.cpp_extension
from torch._inductor import config

sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    pass_message = f"|{name} Test Passed|"
    fail_message = f"|{name} Test Failed|"
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        print("-" * len(pass_message))
        print(pass_message)
        print("-" * len(pass_message))
    else:
        print("-" * len(fail_message))
        print(fail_message)
        print("-" * len(fail_message))

        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        gates = gates.cpu()

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]   # cpu() added to index_sorted_experts to avoid error
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    @torch.compiler.disable(recursive=False)
    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes

        device = inp.device
        inp = inp.cpu()

        inp_exp = inp[self._batch_index].squeeze(1)
        split_tensors = torch.split(inp_exp, self._part_sizes, dim=0)
        split_tensors = tuple(tensor.clone().to(device) for tensor in split_tensors)
        return split_tensors

    @torch.compiler.disable(recursive=True)
    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        expert_out = [out.cpu() for out in expert_out]

        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device="cpu")
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    @torch.compiler.disable(recursive=True)
    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

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

class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

        self.part_sizes = []

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def cv_squared_cpu(self, x):
        device = x.device
        x = x.cpu()
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        result = x.float().var() / (x.float().mean()**2 + eps)
        result = result.to(device)
        return result

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    @torch.compiler.disable(recursive=True)
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean.cpu(), self.std.cpu())
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    @torch.compiler.disable(recursive=True)
    def noisy_top_k_gating_cpu(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        device = x.device
        x = x.cpu()
        w_gate_cpu = self.w_gate.cpu()
        w_noise_cpu = self.w_noise.cpu()

        clean_logits = x @ w_gate_cpu
        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise_cpu
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            torch.manual_seed(0)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits, requires_grad=True) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        # top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_logits, top_indices = self.top_k_cpu(logits, min(self.k + 1, self.num_experts), 1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        # gates = zeros.scatter(1, top_k_indices, top_k_gates)
        gates = self.scatter_cpu(zeros, 1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        device = x.device

        # gates, load = self.noisy_top_k_gating(x, self.training)
        gates, load = self.noisy_top_k_gating_cpu(x, self.training)

        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared_cpu(importance) + self.cv_squared_cpu(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        self.part_sizes.append(dispatcher._part_sizes)  # storing part size info
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs, multiply_by_gates=False)
        return y, loss

    @torch.compiler.disable(recursive=True)
    def scatter_cpu(self, tensor, dim, top_k_indices, top_k_gates):
        tensor = tensor.cpu()
        top_k_gates = top_k_gates.cpu()
        top_k_indices = top_k_indices.cpu()
        return tensor.scatter(dim, top_k_indices, top_k_gates)

    @torch.compiler.disable(recursive=True)
    def top_k_cpu(self, x, k, dim):
        x = x.cpu()
        return x.topk(k, dim)

    @torch.compiler.disable(recursive=True)
    def print_tensors(self, tensors, name=None):
        tensor_name =  name if name is not None else "Tensor"
        if isinstance(tensors, torch.Tensor):
            print(f"{tensor_name}: {tensors.device} : {tensors.to('cpu')}")
        else:
            for i, tensor in enumerate(tensors):
                if isinstance(tensor, torch.Tensor):
                    print(f"{tensor_name} {i} {tensor.device} : {tensor.to('cpu')}")
                else:
                    print(f"{tensor_name} {i} is not a Tensor: {tensor}")

    @torch.compiler.disable(recursive=True)
    def print_weights(self):
        for i, expert in enumerate(self.experts):
            print(f"Expert {i} layer1 weights:\n{expert.fc1.weight.cpu()}")
            print(f"Expert {i} layer2 weights:\n{expert.fc2.weight.cpu()}")


def test_moe(device):
    torch.manual_seed(0)

    # batch_size = 8
    # input_size = 28*28
    # output_size = 8
    # num_experts = 8
    # hidden_size = 32
    # k=2

    # batch_size = 8
    # input_size = 32
    # output_size = 8
    # num_experts = 8
    # hidden_size = 32
    # k=4

    batch_size = 1
    input_size = 16
    output_size=16
    num_experts = 4
    hidden_size=16
    k=2

    # batch_size = 128
    # input_size = 28*28
    # output_size=10
    # num_experts = 8
    # hidden_size=64
    # k=2

    model = MoE(input_size=input_size, output_size=output_size, num_experts=num_experts, hidden_size=hidden_size, k=k, noisy_gating=True)
    model.requres_grad = True
    for i in range(num_experts):
        model.experts[i].requires_grad = True

    model_cpu = copy.deepcopy(model).to("cpu")
    model_cpu.requires_grad = True
    for i in range(num_experts):
        model_cpu.experts[i].requires_grad = True

    X = torch.rand(batch_size, input_size)
    x1 = copy.deepcopy(X).to(device=device)
    x2 = copy.deepcopy(X).to("cpu")

    # model.train()
    model.eval()
    model_device = model.to(device=device)
    opt_model = torch.compile(model_device, dynamic=False)
    y_hat, aux_loss = opt_model(x1)
    print("MoE Custom Device Done!")

    # model_cpu.train()
    model_cpu.eval()
    cpu_hat, cpu_aux_loss = model_cpu(x2)
    test_result("MoE Forward", y_hat, cpu_hat)
    test_result("MoE Aux Loss", aux_loss, cpu_aux_loss)
    print("MoE Forward Done!")

    # Backward
    target = torch.randn(batch_size, output_size)
    y1 = copy.deepcopy(target)
    y2 = copy.deepcopy(target)

    print("Loss Calculation Started!")
    loss = nn.CrossEntropyLoss()(y_hat, y1)
    total_loss = loss + aux_loss
    print("Loss Calculation Done!")

    cpu_loss = nn.CrossEntropyLoss()(cpu_hat, y2)
    total_cpu_loss = cpu_loss + cpu_aux_loss
    total_loss.to(device)

    print("Backward Started!")
    total_loss.backward()
    total_cpu_loss.backward()
    print("MoE Backward Done!")

    # print("MoE Weight Bias print")
    # for i in range(num_experts):
    #     print(f"\nExpert {i}")
    #     print(f"FC1 Weight: {model.experts[i].fc1.weight.cpu()}")
    #     print(f"FC1 Bias: {model.experts[i].fc1.bias.cpu()}")
    #     print("\n")
    #     print(f"FC2 Weight: {model.experts[i].fc2.weight.cpu()}")
    #     print(f"FC2 Bias: {model.experts[i].fc2.bias.cpu()}")
    #     print("\n")

    print("MoE Weight Bias Grad")
    for i in range(num_experts):
        print(f"\nExpert {i}")
        test_result(f"FC1 Grad", model.experts[i].fc1.weight.grad, model_cpu.experts[i].fc1.weight.grad)
        test_result(f"FC1 Bias", model.experts[i].fc1.bias.grad, model_cpu.experts[i].fc1.bias.grad)
        print("\n")
        test_result(f"FC2 Grad", model.experts[i].fc2.weight.grad, model_cpu.experts[i].fc2.weight.grad)
        test_result(f"FC2 Bias", model.experts[i].fc2.bias.grad, model_cpu.experts[i].fc2.bias.grad)
        print("\n")

def train_moe(device):
    def perceptron(a, b, c):
        return a * b + c

    def weight_update(a, b, lr):
        return a - lr * b

    torch.manual_seed(0)
    batch_size = 2
    input_size = 8
    output_size=8
    num_experts = 16
    hidden_size=8
    k=2

    from sklearn.datasets import make_classification
    X, Y = make_classification(n_samples=batch_size, n_features=input_size, n_classes=output_size, n_clusters_per_class=1, n_informative=8, n_redundant=0, n_repeated=0, random_state=0)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    x1 = copy.deepcopy(X).to(device=device)
    x2 = copy.deepcopy(X).to("cpu")

    target = torch.randn(batch_size, output_size)
    y1 = copy.deepcopy(target).to("cpu")
    y2 = copy.deepcopy(target).to("cpu")


    model = MoE(input_size=input_size, output_size=output_size, num_experts=num_experts, hidden_size=hidden_size, k=k, noisy_gating=True)
    model.requres_grad = True
    for i in range(num_experts):
        model.experts[i].requires_grad = True

    model_cpu = copy.deepcopy(model).to("cpu")
    model_cpu.requires_grad = True
    for i in range(num_experts):
        model_cpu.experts[i].requires_grad = True

    model.train()
    # model.eval()
    model_device = model.to(device=device)
    opt_model = torch.compile(model_device, dynamic=False)
    opt_w = torch.compile()(weight_update, dynamic=False)
    y_hat, aux_loss = opt_model(x1)
    print("MoE Custom Device Done!")

    model_cpu.train()
    # model_cpu.eval()
    cpu_hat, cpu_aux_loss = model_cpu(x2)
    test_result("MoE Forward", y_hat, cpu_hat)
    test_result("MoE Aux Loss", aux_loss, cpu_aux_loss)
    print("MoE Forward Done!")

    for i, out in enumerate(y_hat):
        print(f"Expert output {i}, grad_fn: {out.grad_fn}")

    # Loss Calculation
    print("Loss Calculation Started!")
    loss = nn.CrossEntropyLoss()(y_hat, y1)
    total_loss = loss + aux_loss
    cpu_loss = nn.CrossEntropyLoss()(cpu_hat, y2)
    total_cpu_loss = cpu_loss + cpu_aux_loss
    total_loss.to(device)
    print("Loss Calculation Done!")

    # Backward
    print("Backward Started!")
    total_loss.backward()
    total_cpu_loss.backward()
    print("MoE Backward Done!")

    test_result("MoE Forward", y_hat, cpu_hat)
    test_result("MoE Aux Loss", aux_loss, cpu_aux_loss)
    test_result("Loss", total_loss, total_cpu_loss)

    print("MoE Weight Bias Grad")
    for i in range(num_experts):
        print(f"\nExpert {i}")
        test_result(f"FC1 Grad", model.experts[i].fc1.weight.grad, model_cpu.experts[i].fc1.weight.grad)
        test_result(f"FC1 Bias", model.experts[i].fc1.bias.grad, model_cpu.experts[i].fc1.bias.grad)
        print("\n")
        test_result(f"FC2 Grad", model.experts[i].fc2.weight.grad, model_cpu.experts[i].fc2.weight.grad)
        test_result(f"FC2 Bias", model.experts[i].fc2.bias.grad, model_cpu.experts[i].fc2.bias.grad)
        print("\n")

    import matplotlib.pyplot as plt

    # Learning rate
    loss_fn = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = Adam(opt_model.parameters(), lr=lr)
    opt_step = torch.compile(optimizer.step, dynamic=False)
    opt_zero_grad = torch.compile(optimizer.zero_grad, dynamic=False)

    # To record loss values
    loss_values = []
    loop = 5
    # Training loop
    for epoch in range(loop):
        print(f"Epoch {epoch}")

        opt_zero_grad()
        y, aux_loss = opt_model(x1)
        loss = loss_fn(y, y1)
        loss_values.append(loss.item())  # Save loss value
        total_loss = loss + aux_loss
        # total_loss.to(device)
        total_loss.backward()
        optimizer.step()

    # Plotting the loss values
    plt.figure(figsize=(8, 6))
    plt.plot(range(loop), loss_values, label="Training Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('result.png')

def train_moe_mnist(device):
    torch.manual_seed(0)
    batch_size = 32
    input_size = 28*28
    output_size=8
    num_experts = 8
    hidden_size=64
    k=2
    iteration = 10

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
    train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
    num_samples = batch_size * iteration
    indices = [i for i, label in enumerate(train_dataset.targets) if label < 8]
    indices = indices[:num_samples]
    subset_train_mnist = Subset(train_dataset, indices)
    train_loader = DataLoader(dataset=subset_train_mnist, batch_size=batch_size, shuffle=True)

    model = MoE(input_size=input_size, output_size=output_size, num_experts=num_experts, hidden_size=hidden_size, k=k, noisy_gating=True)
    model.requres_grad = True
    for i in range(num_experts):
        model.experts[i].requires_grad = True

    model_cpu = copy.deepcopy(model).to("cpu")
    model_cpu.requires_grad = True
    for i in range(num_experts):
        model_cpu.experts[i].requires_grad = True

    model_device = model.to(device=device)
    opt_model = torch.compile(model_device, dynamic=False)

    loss_fn = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = Adam(opt_model.parameters(), lr=lr)
    opt_step = torch.compile(optimizer.step, dynamic=False)
    opt_zero_grad = torch.compile(optimizer.zero_grad, dynamic=False)

    def train(model, device, train_loader, optimizer, epochs):
        model.train()
        loss_list = []
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.view(data.size(0), -1).to(device), target.to(device)
                # optimizer.zero_grad()
                opt_zero_grad()
                print(f"Feeding data shape {data.shape}")
                output, aux_loss = model(data)
                loss = loss_fn(output, target)
                total_loss = loss + aux_loss
                total_loss.backward()
                # optimizer.step()
                opt_step()

                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.cpu():.6f}')
                loss_list.append(loss.cpu().detach())
        return loss_list

    epochs = 10
    loss_list = train(opt_model, device, train_loader, optimizer, epochs)

    name = f"moe_{batch_size}_{input_size}_{output_size}_{num_experts}_{hidden_size}_{k}_{iteration}_{epochs}"
    # dump loss_list to a file
    with open(f'{name}_loss_list.txt', 'w') as f:
        for item in loss_list:
            f.write("%s\n" % item)

    # Plotting the loss values
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs * iteration), loss_list, label="Training Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'{name}_result.png')

def train_moe_single_iteration(device, iter_idx, is_evaluation=0):
    # Training moe with mnist dataset for sinlge iteration
    torch.manual_seed(0)
    batch_size = 128
    input_size = 28*28
    output_size=10
    num_experts = 8
    hidden_size=64
    k=2

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
    train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
    # num_samples = batch_size * iteration
    indices = [i for i, label in enumerate(train_dataset.targets)]
    # indices = indices[:num_samples]
    iteration = len(indices) // batch_size
    indice_idx = iter_idx % iteration
    indices = indices[batch_size * indice_idx:batch_size * (indice_idx + 1)]
    subset_train_mnist = Subset(train_dataset, indices)
    train_loader = DataLoader(dataset=subset_train_mnist, batch_size=batch_size, shuffle=True)
    subsete_test_mnist = Subset(test_dataset, indices)
    evaluation_loader = DataLoader(dataset=subsete_test_mnist, batch_size=batch_size, shuffle=True)

    model = MoE(input_size=input_size, output_size=output_size, num_experts=num_experts, hidden_size=hidden_size, k=k, noisy_gating=True)
    model.requres_grad = True
    # for i in range(num_experts):
    #     model.experts[i].requires_grad = True

    # load weight from path
    if iter_idx > 0:
        path = f"./params/{iter_idx - 1}.pt"
        print(f"Loading model from {path}")
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
            print(f"Model loaded from {path}")

    model_device = model.to(device=device)
    opt_model = torch.compile(model_device, dynamic=False)

    loss_fn = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = Adam(opt_model.parameters(), lr=lr)
    opt_step = torch.compile(optimizer.step, dynamic=False)
    opt_zero_grad = torch.compile(optimizer.zero_grad, dynamic=False)


    def train(opt_model, train_loader):
        print("Training Model")
        opt_model.train()

        data, target = next(iter(train_loader))
        data, target = data.view(data.size(0), -1).to(device), target.to(device)

        opt_zero_grad()
        print(f"Feeding data shape {data.shape}")
        output, aux_loss = opt_model(data)
        loss = loss_fn(output, target)
        total_loss = loss + aux_loss
        total_loss.backward()
        opt_step()

        print(f"Train {iter_idx}: Loss: {loss.cpu().detach()}")

        loss = loss.cpu().detach()
        # save model
        path = f"./params/{iter_idx}.pt"
        # torch.save(model.state_dict(), path)
        torch.save({key: value.cpu() for key, value in model.state_dict().items()}, path)

        # append lost to file
        path = f"./params/loss.txt"
        with open(path, 'a') as f:
            f.write(f"{loss}\n")

        path = f"./params/part_sizes.txt"
        with open(path, 'a') as f:
            part_size = model.part_sizes[0]
            f.write(f"{part_size}\n")

        # return loss.cpu().detach()

    def evaluation(model, evaluation_loader):
        print("Evaluation Model")
        evaluation_loss = 0
        evaluation_total = 0
        evaluation_correct = 0
        model.eval()
        with torch.no_grad():
            data, target = next(iter(evaluation_loader))
            data, target = data.view(data.size(0), -1).to(device), target.to(device)
            output, aux_loss = model(data)
            loss = loss_fn(output, target)
            evaluation_loss += loss.cpu().detach()
            _, predicted = torch.max(output, 1)
            evaluation_total += target.size(0)
            evaluation_correct += (predicted == target).sum().item()
            print(f"evaluation {iter_idx}: Loss: {loss.cpu().detach()}")

        # store evaluation result to path
        path = f"./params/evaluation_loss.txt"
        line = f"iter: {iter_idx}, loss: {evaluation_loss}, correct: {evaluation_correct}, total: {evaluation_total}\n"
        with open(path, 'a') as f:
            f.write(line)


    if is_evaluation:
        evaluation(opt_model, evaluation_loader)
    else:
        train(opt_model, train_loader)

if __name__ == "__main__":
    from Scheduler.scheduler import ExecutionEngine
    torch.set_printoptions(threshold=float('inf'), linewidth=600)
    module = ExecutionEngine.setup_device()
    device = module.custom_device()

    test_moe(device)
    # train_moe(device)
    # train_moe_mnist(device)
