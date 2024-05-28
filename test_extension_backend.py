# Owner(s): ["module: inductor"]
import os
import shutil
import sys
import time
import contextlib
import unittest
import numpy as np

import torch
import torch.nn as nn
import torch._dynamo
import torch.utils.cpp_extension
from torch._inductor import config

try:
    from PyTorchSimFrontend.llvm_codegen_backend import (
        MatrixLLVMScheduling,
        ExtensionWrapperCodegen,
    )
except ImportError:
    from .PyTorchSimFrontend.llvm_codegen_backend import (
        MatrixLLVMScheduling,
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
            "extension_device", MatrixLLVMScheduling, ExtensionWrapperCodegen
        )
        self.assertTrue(
            get_scheduling_for_device("extension_device") == MatrixLLVMScheduling
        )
        self.assertTrue(
            get_wrapper_codegen_for_device("extension_device")
            == ExtensionWrapperCodegen
        )

        self.assertFalse(self.module.custom_op_called())
        device = self.module.custom_device()
        x = torch.empty(1000, 1000).to(device=device).fill_(1)
        self.assertTrue(self.module.custom_op_called())
        y = torch.empty(1000, 1000).to(device=device).fill_(2)
        z = torch.empty(1000).to(device=device).fill_(3)
        ref = torch.empty(1000, 1000).fill_(3)

        self.assertTrue(x.device == device)
        self.assertTrue(y.device == device)
        self.assertTrue(z.device == device)

        def fn(a, b, c):
            return a * b + c

        def vectoradd(a, b):
            return a + b

        def reduce_sum(a, b):
            return torch.sum(a + b, axis=-1)

        def custom_matmul(a, b):
            return torch.matmul(a, b)

        def custom_conv2d(a, b):
            i_c = a.shape[1]
            o_c = b.shape[0]
            conv2d = nn.Conv2d(i_c, o_c, b.shape[-1], stride=1, padding=0, dilation=1)
            conv2d.weight = nn.Parameter(b)

            return conv2d(a)

        metrics.reset()
        # conv test
        input_array = np.arange(1*8*64*64)*0.001
        conv_input = torch.tensor(input_array, dtype=torch.float32).view(1, 8, 64, 64).to(device=device)
        kernel_array = np.arange(16*8*3*3)*0.001
        conv_kernel = torch.tensor(kernel_array, dtype=torch.float32).view(16, 8, 3, 3).to(device=device)

        opt_fn = torch.compile()(custom_conv2d)
        res = opt_fn(conv_input, conv_kernel)
        # self.assertEqual(ref, res.to(device="cpu"))

        out = custom_conv2d(conv_input.cpu(), conv_kernel.cpu())

        print("Result > ", torch.allclose(res.cpu(), out, rtol=1, atol=1))
        print("Max diff > ", torch.max(torch.abs(res.cpu() - out)))

        # Backward Uni-Test (Single Perceptron)
        def perceptron(a, b, c):
            res = a * b + c
            return res.sum()

        def weight_update(a, b, lr):
            return a - b * lr

        x = torch.empty(64).to(device=device).fill_(1)
        w = torch.empty(64).to(device=device).fill_(3)
        target_y = torch.tensor(384, dtype=torch.float32).to(device=device)
        b = torch.empty(64).to(device=device).fill_(0.1)
        w.requires_grad = True
        b.requires_grad = True
        opt_mlp = torch.compile()(perceptron)
        opt_w = torch.compile()(weight_update)
        lr = torch.tensor(1e-3).to(device=device) # learning rate
        for i in range(50):
            y = opt_mlp(x, w, b)
            loss = (target_y - y) ** 2 # TODO: Add loss function
            # print(loss.cpu().item()) # check loss
            # loss.to(device=device)
            loss.backward()
            with torch.no_grad():
                w.copy_(opt_w(w, w.grad, lr))
                b.copy_(opt_w(b, b.grad, lr))
            w.grad.zero_()
            b.grad.zero_()

        # GEMM Backward TEST
        torch.manual_seed(0)
        input = torch.randn(64, 64)
        weight = torch.randn(64, 64)

        x1 = input.to(device=device)
        w1 = weight.to(device=device)
        x2 = input.to("cpu")
        w2 = weight.to("cpu")
        self.assertTrue(x1.device == device)
        self.assertTrue(w1.device == device)
        self.assertTrue(x2.device == torch.device("cpu"))
        self.assertTrue(w2.device == torch.device("cpu"))
        w1.requires_grad = True
        opt_fn = torch.compile()(custom_matmul)
        res = opt_fn(x1, w1)
        loss = torch.sum(res)
        loss.backward()
        print(w1.grad.cpu())
        w2.requires_grad = True
        y = custom_matmul(x2, w2)
        loss2 = torch.sum(y)
        loss2.backward()
        print(w2.grad.cpu())

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    from torch.testing._internal.inductor_utils import HAS_CPU

    if HAS_CPU and not IS_MACOS:
        run_tests(needs="filelock")