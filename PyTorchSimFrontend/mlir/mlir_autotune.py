import functools
import torch
import os
import dataclasses
from torch._inductor.autotune_process import TensorMeta
from torch._inductor.codecache import get_hash, write
from PyTorchSimFrontend import extension_config
from Simulator.simulator import TOGSimulator

from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Union,
)

# FIXME. Avoid circular import
def hash_prefix(hash_value):
    return hash_value[1:12]

def get_write_path(src_code):
    return os.path.join(extension_config.CONFIG_TORCHSIM_DUMP_PATH, "outputs", hash_prefix(get_hash(src_code.strip())))

@dataclasses.dataclass
class MLIRBenchmarkRequest():
    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        extra_args: Iterable[Any],
        source_code: str,
    ):
        self.kernel_name = kernel_name
        if isinstance(input_tensor_meta, TensorMeta):
            input_tensor_meta = [input_tensor_meta]
        self.input_tensor_meta = input_tensor_meta

        if isinstance(output_tensor_meta, TensorMeta):
            output_tensor_meta = [output_tensor_meta]
        self.output_tensor_meta = output_tensor_meta
        self.source_code = source_code
        self.workspace_size: int = 0
        self.workspace: Optional[torch.Tensor] = None
        self.hash_key: str = ""
        self.source_file: str = ""
        self.extra_args = extra_args
        #self.hash_key, self.source_file = CUDACodeCache.write(self.source_code, "so")

    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.source_file=}, {self.hash_key=}"

    def make_run_fn(
        self, input_tensors: torch.Tensor, output_tensors: torch.Tensor
    ) -> Callable[[], None]:
        from PyTorchSimFrontend.extension_codecache import CustomAsyncCompile
        custom_async_compile = CustomAsyncCompile()

        # Check already cached result.
        write_path = get_write_path(self.source_code)
        key,  _ = write(self.source_code, "mlir", specified_dir=write_path)
        result_dir = os.path.join(extension_config.CONFIG_TORCHSIM_DUMP_PATH, "outputs", hash_prefix(key), "togsim_result")

        # Find the most recent .log file in the result directory
        if os.path.exists(result_dir) and os.path.isdir(result_dir):
            log_files = [f for f in os.listdir(result_dir) if f.endswith('.log')]
            if log_files:
                # Sort by modification time, get the most recent file
                log_files_with_time = [
                    (f, os.path.getmtime(os.path.join(result_dir, f)))
                    for f in log_files
                ]
                log_files_with_time.sort(key=lambda x: x[1], reverse=True)
                latest_log_file = log_files_with_time[0][0]
                result_path = os.path.join(result_dir, latest_log_file)
                result = TOGSimulator.get_result_from_file(result_path)
                def cached_run_fn(*args, **kwargs):
                    return result
                return cached_run_fn

        # Run a candidate code
        run_method = custom_async_compile.mlir(
            self.source_code, vectorlane_size=self.extra_args["vector_lane"],
            loop_size=None, spad_info=self.extra_args["spad_info"],
            vlen=self.extra_args["vlen"], arg_attributes=self.extra_args["arg_attributes"],
            origins=self.extra_args["origins"], silent_mode=True,
            autotune=self.extra_args['autotune'])

        args = [
            tensor
            for tensor in list(input_tensors) + list(output_tensors)
        ]

        # Generate partial function.
        return functools.partial(
            run_method,
            *args,
        )

    def update_workspace_size(self) -> None:
        # FIXME: Not implemented yet. Checkout torch/_inductor/codegen/rocm/rocm_benchmark_request.py
        return