import functools
import torch
from torch._inductor.autotune_process import BenchmarkRequest
from torch._inductor.autotune_process import TensorMeta
from torch._inductor.codecache import CUDACodeCache

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Union,
)

class LLVMBenchmarkRequest(BenchmarkRequest):
    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        extra_args: Iterable[Any],
        source_code: str,
    ):
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.workspace_size: int = 0
        self.workspace: Optional[torch.Tensor] = None
        self.hash_key: str = ""
        self.source_file: str = ""
        #self.hash_key, self.source_file = CUDACodeCache.write(self.source_code, "so")

    def make_run_fn(
        self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
    ) -> Callable[[], None]:
        self.DLL, self.hash_key, self.source_file = CUDACodeCache.load(
            self.source_code, "so"
        )

        args = [
            tensor.data_ptr()
            for tensor in list(input_tensors) + [output_tensor]
        ]

        print(
            "make_run_fn: self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.source_file,
            self.hash_key,
            args,
            self.extra_args,
        )

        run_method = getattr(self.DLL, self.kernel_name)

        # Retrieve workspace_size and initialize workspace.
        run_method(
            *args,  # input ptrs and output ptrs
            *self.extra_args,
        )

        # Generate partial function.
        return functools.partial(
            run_method,
            *args,
            *self.extra_args,
            None,  # null workspace size ptr
            None,  # set workspace ptr, TODO: update it to a real ptr if workspace_size > 0
        )

    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.source_file=}, {self.hash_key=}"