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

class MLIRBenchmarkRequest(BenchmarkRequest):
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
        from PyTorchSimFrontend.extension_codecache import CustomAsyncCompile
        custom_async_compile = CustomAsyncCompile()
        run_method = custom_async_compile.mlir(
            self.source_code, vectorlane_size=self.extra_args["vector_lane"],
            loop_size=None, spad_info=self.extra_args["spad_info"],
            vlen=self.extra_args["vlen"], arg_attributes=self.extra_args["arg_attributes"],
            origins="Unknown")

        args = [
            tensor
            for tensor in list(input_tensors) + [output_tensor]
        ]

        # Generate partial function.
        return functools.partial(
            run_method,
            *args,
        )

    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.source_file=}, {self.hash_key=}"