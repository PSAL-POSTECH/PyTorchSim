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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch._dynamo
import torch.utils.cpp_extension
from torch._inductor import config

try:
    from PyTorchSimFrontend.mlir.mlir_codegen_backend import (
        MLIRScheduling,
        ExtensionWrapperCodegen,
    )
except ImportError:
    from .PyTorchSimFrontend.mlir.mlir_codegen_backend import (
        MLIRScheduling,
        ExtensionWrapperCodegen,
    )

from torch._C import FileCheck
from torch._inductor import metrics
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
    register_backend_for_device,
)
from torch.testing._internal.common_utils import IS_MACOS
from torch.testing._internal.common_utils import TestCase as TorchTestCase

def remove_build_path():
    if sys.platform == "win32":
        # Not wiping extensions build folder because Windows
        return
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        shutil.rmtree(default_build_root, ignore_errors=True)

class TestCase(TorchTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "debug": True,
                    "debug_index_asserts": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                    "generate_intermediate_hooks": True,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        super().setUp()
        self._start = time.perf_counter()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()
        if os.environ.get("ERROR_ON_SLOW") == "1":
            elapsed = time.perf_counter() - self._start
            assert elapsed < 120


class ExtensionBackendTests(TestCase):
    module = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Build Extension
        remove_build_path()
        source_file_path = os.path.dirname(os.path.abspath(__file__))
        source_file = os.path.join(
            source_file_path, "PyTorchSimFrontend/extension_device.cpp"
        )
        cls.module = torch.utils.cpp_extension.load(
            name="extension_device",
            sources=[
                str(source_file),
            ],
            extra_cflags=["-g"],
            verbose=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

        remove_build_path()

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        assert self.module is not None

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

        # return the working directory (see setUp)
        os.chdir(self.old_working_dir)

    def test_open_device_registration(self):
        torch.utils.rename_privateuse1_backend("extension_device")

        register_backend_for_device(
            "extension_device", MLIRScheduling, ExtensionWrapperCodegen
        )
        self.assertTrue(
            get_scheduling_for_device("extension_device") == MLIRScheduling
        )
        self.assertTrue(
            get_wrapper_codegen_for_device("extension_device")
            == ExtensionWrapperCodegen
        )

class MLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=64, output_size=8):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # x = self.relu(x)
        # x = self.softmax(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.norm = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        # x = self.conv2(x)
        # x = self.maxpool(x)
        # x = torch.nn.functional.relu(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, n):
        super(LayerNorm, self).__init__()
        self.ln = nn.LayerNorm(n)

    def forward(self, x):
        return self.ln(x)

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        return multihead_attn(x, x, x)

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DecoderBlock, self).__init__()
        self.multihead_attn = my_MultiheadAttention(num_heads, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = nn.Linear(embed_dim, embed_dim*3)
        self.act = nn.ReLU()
        self.ffn2 = nn.Linear(embed_dim*3, embed_dim)

    def forward(self, x):
        result = self.multihead_attn(x, x, x)
        result = self.layer_norm(result+x)

        ffn1_result = self.ffn1(result)
        act_result = self.act(ffn1_result)
        ffn2_result = self.ffn2(act_result)
        return self.layer_norm(ffn2_result + result)

import math
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class my_MultiheadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(my_MultiheadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None

    def attention(self, query, key, value):
        d_k = query.size(-1)
        print("Attention in CPU >")
        print("Score CPU > ")
        print(torch.matmul(query, key.transpose(-2, -1)))

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = scores.softmax(dim=-1)
        print("Softmax CPU > ")
        print(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value):
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(-1, self.h, self.d_k).transpose(0, 1).contiguous()
            for lin, x in zip(self.linears, (query, key, value))
        ]

        if query.device == torch.device("cpu"):
            print("QKV After Linear Projection in CPU >")
            print("CPU Query > ", query)
            print("CPU Query Weight > ", self.linears[0].weight)
            print("CPU Query Bias > ", self.linears[0].bias)
            print("CPU Key > ", key)
            print("CPU Key Weight > ", self.linears[1].weight)
            print("CPU Key Bias > ", self.linears[1].bias)
            print("CPU Value > ", value)
            print("CPU Value Weight > ", self.linears[2].weight)
            print("CPU Value Bias > ", self.linears[2].bias)

        # 2) Apply attention on all the projected vectors in batch.
        # x, self.attn = self.attention(query, key, value)
        # d_k = query.size(-1)

        if query.device == torch.device("cpu"):
            print("Attention in CPU >")
            print("Score CPU > ")
            print(torch.matmul(query, key.transpose(-2, -1)))

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        p_attn = scores.softmax(dim=-1)
        if query.device == torch.device("cpu"):
            print("Softmax CPU > ")
            print(p_attn)
        x = torch.matmul(p_attn, value)
        if query.device == torch.device("cpu"):
            print("Attention Result in CPU >")
            print("CPU X > ", x)

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(0, 1)
            .contiguous()
            .view(-1, self.h * self.d_k)
        )

        if query.device == torch.device("cpu"):
            print("X After Concat in CPU >")
            print("CPU X > ", x)

        del query
        del key
        del value
        return self.linears[-1](x)

class my_Decoder(nn.Module):
    def __init__(self):
        # custom transformer decoder
        super(my_Decoder, self).__init__()

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    message = f"|{name} Test Passed|"
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)

def test_vectoradd(device, size=(128, 128)):
    def vectoradd(a, b):
        return a + b
    x = torch.randn(size).to(device=device)
    y = torch.randn(size).to(device=device)
    opt_fn = torch.compile()(vectoradd)
    res = opt_fn(x, y)
    out = vectoradd(x.cpu(), y.cpu())
    test_result("VectorAdd", res, out)

def test_reduce_sum(device, size, dim, keepdim=False):
    def reduce_sum(a, b, dim, keepdim):
        return torch.sum(a + b, axis=dim, keepdim=keepdim)
    x = torch.randn(size).to(device=device)
    y = torch.randn(size).to(device=device)
    opt_fn = torch.compile()(reduce_sum)
    res = opt_fn(x, y, dim, keepdim)
    out = reduce_sum(x.cpu(), y.cpu(), dim, keepdim)
    test_result("ReduceSum", res, out)

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
    opt_mlp = torch.compile()(perceptron)
    opt_w = torch.compile()(weight_update)
    opt_loss = torch.compile()(torch.nn.MSELoss())
    lr = torch.tensor(5e-2).to(device=device) # learning rate
    y = opt_mlp(w1, x1, b1)
    loss = opt_loss(y, y1)
    loss.backward()
    cpu_y = perceptron(x2, w2, b2)
    cpu_loss = torch.nn.MSELoss()(cpu_y, y2)
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

def test_matmul(device, input_size=128, hidden_size=128, output_size=128):
    def custom_matmul(bias, a, b):
        return torch.addmm(bias, a, b)
    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    bias = torch.randn(output_size)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    b1 = bias.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    b2 = bias.to("cpu")
    opt_fn = torch.compile()(custom_matmul)
    res = opt_fn(b1, x1, w1)
    y = custom_matmul(b2, x2, w2)
    test_result("Matmul Forward", res, y)

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
    opt_fn = torch.compile()(model)
    y = opt_fn(x1)
    cpu_model = copy.deepcopy(model).to("cpu")
    cpu_model.requires_grad = True
    cpu_y = cpu_model(x2)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt_loss = torch.compile()(loss_fn)
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

def test_CNN(device):
    torch.manual_seed(0)
    input = torch.randn(1, 8, 64, 64)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    model = CNN().eval()
    model.to(device=device)
    opt_fn = torch.compile()(model)
    y = opt_fn(x1)
    cpu_model = model.to("cpu")
    cpu_y = cpu_model(x2)
    test_result("CNN Forward", y, cpu_y, rtol=2e-1, atol=2e-1)
    print("Max diff > ", torch.max(torch.abs(y.cpu() - cpu_y)))

def test_conv2d(device):
    def custom_conv2d(a, b):
        i_c = a.shape[1]
        o_c = b.shape[0]
        conv2d = nn.Conv2d(i_c, o_c, b.shape[-1], stride=1, padding=0, dilation=1)
        conv2d.weight = nn.Parameter(b)
        return conv2d(a)
    torch.manual_seed(0)
    conv_input = torch.randn(1, 8, 64, 64).to(device=device)
    conv_kernel = torch.randn(16, 8, 3, 3).to(device=device)
    opt_fn = torch.compile()(custom_conv2d)
    res = opt_fn(conv_input, conv_kernel)
    out = custom_conv2d(conv_input.cpu(), conv_kernel.cpu())
    test_result("Conv2d Forward", res, out, rtol=1e-1, atol=1e-1)
    print("Max diff > ", torch.max(torch.abs(res.cpu() - out)))

def test_softmax(device, size=(128, 128), dim=1):
    torch.manual_seed(0)
    input = torch.randn(size)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    opt_fn = torch.compile()(torch.nn.functional.softmax)
    y = opt_fn(x1, dim=dim)
    cpu_y = torch.nn.functional.softmax(x2, dim=dim)
    test_result("Softmax", y, cpu_y)

def test_ReLU(device, size=(128, 128)):
    torch.manual_seed(0)
    input = torch.randn(size)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    opt_fn = torch.compile()(torch.nn.functional.relu)
    y = opt_fn(x1)
    cpu_y = torch.nn.functional.relu(x2)
    test_result("ReLU", y, cpu_y)

def test_LayerNorm(device, size=(64, 64)):
    torch.manual_seed(0)
    input = torch.randn(size)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    model = LayerNorm(size[-1])
    model.to(device=device)
    opt_fn = torch.compile()(model)
    y = opt_fn(x1)
    cpu_model = model.to("cpu")
    cpu_y = cpu_model(x2)
    test_result("LayerNorm Forward", y, cpu_y)

def test_BatchNorm(device, size=(1, 16, 64, 64)):
    torch.manual_seed(0)
    model = nn.BatchNorm2d(size[1]).eval()
    model.to(device=device)
    input = torch.randn(size)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    opt_fn = torch.compile()(model)
    y = opt_fn(x1)
    cpu_model = model.to("cpu")
    cpu_y = cpu_model(x2)
    test_result("BatchNorm Forward", y, cpu_y)

def test_Attention(device):
    def attention(query, key, value):
        import math
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = scores.softmax(dim=-1)
        return torch.matmul(p_attn, value), p_attn

    torch.manual_seed(0)
    query = torch.randn(16, 128).to(device=device)
    key = torch.randn(16, 128).to(device=device)
    value = torch.randn(16, 128).to(device=device)

    opt_fn = torch.compile()(attention)
    res, p_attn = opt_fn(query, key, value)

    cpu_res, cpu_p_attn = attention(query.cpu(), key.cpu(), value.cpu())
    test_result("Attention Forward", res, cpu_res)

def test_MultiAttention(device):
    torch.manual_seed(0)
    cpu_query = torch.randn(16, 128)
    cpu_key = torch.randn(16, 128)
    cpu_value = torch.randn(16, 128)

    query = cpu_query.clone().to(device=device)
    key = cpu_key.clone().to(device=device)
    value = cpu_value.clone().to(device=device)

    #print("Model Input Print >>>>>> ")
    #print("Query > ", query.cpu())
    #print("Key > ", key.cpu())
    #print("Value > ", value.cpu())

    multihead_attn = my_MultiheadAttention(8, 128)
    cpu_multihead_attn = multihead_attn.to("cpu")
    cpu_res = cpu_multihead_attn(cpu_query, cpu_key, cpu_value)
    multihead_attn.to(device=device)
    opt_fn = torch.compile()(multihead_attn)
    res = opt_fn(query, key, value)

    test_result("Multihead Attention Forward", res, cpu_res)

def test_DecoderBlock(device):
    cpu_query = torch.randn(16, 128)
    decoder_block = DecoderBlock(128, 8)
    cpu_res = decoder_block(cpu_query)

    query = cpu_query.clone().to(device=device)
    decoder_block.to(device=device)
    opt_fn = torch.compile()(decoder_block)
    res = opt_fn(query)

    test_result("Decoder Block Forwrad", res, cpu_res)

def test_BMM(device):
    def bmm(a, b):
        return torch.bmm(a, b.transpose(1, 2))
    torch.manual_seed(0)
    a = torch.randn(1, 32, 64).to(device=device)
    b = torch.randn(1, 16, 64).to(device=device)
    opt_fn = torch.compile()(bmm)
    res = opt_fn(a, b)
    out = bmm(a.cpu(), b.cpu())
    test_result("BMM Forward", res, out)

def test_Transpose2D(device, size=(16, 32)):
    def transpose(a):
        return a.transpose(0, 1).contiguous()
    torch.manual_seed(0)
    # x = torch.randn(16, 32).to(device=device)
    x = torch.randn(size[0], size[1]).float().to(device=device)
    opt_fn = torch.compile()(transpose)
    res = opt_fn(x)
    out = transpose(x.cpu())
    test_result("Transpose Forward", res, out)

def test_Transpose2D_2(device, size=(16, 32)):
    def transpose(a, b):
        return a.transpose(0, 1) + b
    torch.manual_seed(0)
    # x = torch.randn(16, 32).to(device=device)
    x = torch.randn(size[0], size[1]).float().to(device=device)
    y = torch.randn(size[1], size[0]).float().to(device=device)

    opt_fn = torch.compile()(transpose)
    res = opt_fn(x, y)
    out = transpose(x.cpu(), y.cpu())
    test_result("Transpose2 Forward", res, out)

def test_Transpose3D_1(device, size=(4, 16, 32)):
    def transpose(a, b):
        return a.transpose(1, 2) + b
    torch.manual_seed(0)
    # x = torch.randn(16, 32).to(device=device)
    x = torch.randn(size[0], size[2], size[1]).float().to(device=device)
    y = torch.randn(size[0], size[1], size[2]).float().to(device=device)

    opt_fn = torch.compile()(transpose)
    res = opt_fn(x, y)
    out = transpose(x.cpu(), y.cpu())
    test_result("Transpose 3D Forward", res, out)

def test_Transpose3D_2(device, size=(4, 16, 32)):
    def transpose(a, b):
        return a.transpose(0, 2) + b
    torch.manual_seed(0)
    # x = torch.randn(16, 32).to(device=device)
    x = torch.randn(size[2], size[1], size[0]).float().to(device=device)
    y = torch.randn(size[0], size[1], size[2]).float().to(device=device)

    opt_fn = torch.compile()(transpose)
    res = opt_fn(x, y)
    out = transpose(x.cpu(), y.cpu())
    test_result("Transpose 3D Forward", res, out)

def test_Transpose3D_3(device, size=(4, 16, 32)):
    def transpose(a, b):
        return a.transpose(0, 1) + b
    torch.manual_seed(0)
    # x = torch.randn(16, 32).to(device=device)
    x = torch.randn(size[1], size[0], size[2]).float().to(device=device)
    y = torch.randn(size[0], size[1], size[2]).float().to(device=device)

    opt_fn = torch.compile()(transpose)
    res = opt_fn(x, y)
    out = transpose(x.cpu(), y.cpu())
    test_result("Transpose 3D Forward", res, out)

def MLP_MNIST(device):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
    train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
    indices = [i for i, label in enumerate(train_dataset.targets) if label < 8]
    subset_train_mnist = Subset(train_dataset, indices)
    train_loader = DataLoader(dataset=subset_train_mnist, batch_size=964, shuffle=True)

    model = MLP(input_size=28*28, hidden_size=64, output_size=8).to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    opt_model = torch.compile()(model)
    opt_step = torch.compile()(optimizer.step)
    opt_zero_grad = torch.compile()(optimizer.zero_grad)

    def train(model, device, train_loader, optimizer, epochs):
        model.train()
        loss_list = []
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.view(data.size(0), -1).to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                opt_zero_grad()
                loss.backward()
                opt_step()
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.cpu():.6f}')
                loss_list.append(loss.cpu().detach())
        return loss_list

    loss_list = train(opt_model, device, train_loader, optimizer, 2)
    import csv
    with open('mlp_loss.csv', 'w') as f:
        writer = csv.writer(f)
        for loss in loss_list:
            writer.writerow([loss.item()])

def test_optimizer(device):
    torch.manual_seed(0)
    model = MLP(input_size=16, hidden_size=16, output_size=16).to(device=device)
    cpu_model = copy.deepcopy(model).to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cpu_optimizer = torch.optim.Adam(cpu_model.parameters(), lr=0.001)
    opt_step = torch.compile()(optimizer.step)
    input = torch.randn(16, 16)
    x1 = copy.deepcopy(input).to(device=device)
    x2 = copy.deepcopy(input).to("cpu")
    y = model(x1)
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

def test_maxpool(device):
    torch.manual_seed(0)
    model = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).eval()
    model.to(device=device)
    input = torch.randn(1, 8, 64, 64).to(device=device)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    opt_fn = torch.compile()(model)
    res = opt_fn(x1)
    model.to("cpu")
    out = model(x2)
    # test_result("Maxpool Forward", res, out) # TODO: MaxPool Functionality is not working

def test_avgpool(device):
    def avgpool(a):
        return nn.AdaptiveAvgPool2d((1, 1))(a)
    torch.manual_seed(0)
    input = torch.randn(1, 16, 64, 64).to(device=device) #FIXME: channel 8 does not work (range padding issue)
    x1 = input.to(device=device)
    x2 = input.to("cpu")
    opt_fn = torch.compile()(avgpool)
    res = opt_fn(x1)
    out = avgpool(x2)
    test_result("Avgpool Forward", res, out)

def test_view3D_2D(device):
    def view3D_2D(a):
        return a.view(16, 128).contiguous()
    torch.manual_seed(0)
    cpu_input = torch.randn(16, 8, 16)
    input = cpu_input.clone().to(device=device)
    opt_fn = torch.compile()(view3D_2D)
    res = opt_fn(input)
    out = view3D_2D(input)
    test_result("view 3D->2D", res, out)

def test_moe(device):
    from moe import MoE
    model = MoE(input_size=1000, output_size=20, num_experts=10, hidden_size=66, k= 4, noisy_gating=True)
    X = torch.rand(32, 1000)
    x1 = X.to(device=device)
    x2 = X.to("cpu")
    model.eval()
    cpu_model = copy.deepcopy(model).to("cpu")
    model = model.to(device=device)
    opt_model = torch.compile(model)
    y_hat, aux_loss = opt_model(x1)
    cpu_hat, cpu_aux_loss = cpu_model(x2)
    test_result("MoE Forward", y_hat, cpu_hat)
    test_result("MoE Aux Loss", aux_loss, cpu_aux_loss)

def test_resnet(device):
    from torchvision.models import resnet18
    model = resnet18().eval()
    model.to(device)
    input = torch.randn(1, 3, 224, 224).to(device=device)
    x1 = input.to(device=device)
    opt_fn = torch.compile()(model)
    res = opt_fn(x1)
    print("ResNet18 Simulation Done")

def test_matmul_fused(device):
    def matmul_fused(a, b, c):
        return torch.matmul(a, b) * c
    torch.manual_seed(0)
    input = torch.randn(128, 128)
    weight = torch.randn(128, 128)
    bias = torch.randn(128)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    c = 7
    opt_fn = torch.compile()(matmul_fused)
    res = opt_fn(x1, w1, c)
    y = matmul_fused(x2, w2, c)
    test_result("Matmul Forward", res, y)

if __name__ == "__main__":
    #from torch._dynamo.test_case import run_tests
    #from torch.testing._internal.inductor_utils import HAS_CPU
    #if HAS_CPU and not IS_MACOS:
    #    run_tests(needs="filelock")

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    # torch.set_printoptions(threshold=float('inf'), linewidth=600)
    test_vectoradd(device, (47, 10))
    test_reduce_sum(device, (29, 47), 1, keepdim=True)
    test_reduce_sum(device, (16, 64), 0, keepdim=True)
    test_Transpose2D(device, [64, 64])
    test_Transpose2D_2(device, [16, 64])
    test_Transpose3D_1(device, [64, 64, 64])
    test_Transpose3D_2(device, [64, 64, 64])
    test_Transpose3D_3(device, [64, 64, 64])
    test_view3D_2D(device)
    test_maxpool(device)
    test_avgpool(device)
    test_softmax(device, (64, 128), dim=1)
    test_BatchNorm(device)
    test_matmul(device)
    test_CNN(device)
    test_DecoderBlock(device)
    test_resnet(device)
    test_mlp(device)

    test_BMM(device)
    test_LayerNorm(device, (64, 128))
    test_conv2d(device)

    # # Fusion Test
    test_matmul_fused(device)