import torch
from PyTorchSimFrontend.mlir.mlir_common import MLIRKernelArgs
from PyTorchSimFrontend.llvm.llvm_caller_codegen import LLVMKernelCallerCodeGen
from PyTorchSimFrontend.mlir.mlir_common import DTYPE_TO_C

class MLIRKernelCallerCodeGen(LLVMKernelCallerCodeGen):

    def __init__(self, validation, arg_attributes, cycle_sim=False):
        super().__init__(validation, arg_attributes)
        self.cycle_sim = cycle_sim

    def write_header(self):
        super().write_header()
        global_var_header = "gem5_global_var.h" if self.cycle_sim else "global_var.h"
        self.writeline(f"#include \"{global_var_header}\"")

    def is_in_arg(self, value):
        return MLIRKernelArgs.is_mlir_arg_in(value)

    def is_out_arg(self, value):
        return MLIRKernelArgs.is_mlir_arg_out(value)

    def is_inout_arg(self, value):
        return MLIRKernelArgs.is_mlir_arg_inout(value)

    def is_var_arg(self, value):
        return MLIRKernelArgs.is_mlir_arg_var(value)

    def load_arg(self):
        for arg_name, arg_attribute in self.arg_attributes:
            if self.is_in_arg(arg_attribute[0]) or self.is_var_arg(arg_attribute[0]):
                argv_idx = self.load_args[arg_name]
                argv_sz_idx = self.load_args[arg_name + "_sz"]
                self.load_args[arg_name] = argv_idx
                self.writeline(f'if(load_arg(c_{arg_name}, atoi(argv[{argv_sz_idx}]), argv[{argv_idx}]) == -1){self.open_bracket}')
                with self.code.indent():
                    self.writeline(f'return -1{self.ending}')
                self.writeline(self.closed_bracket)
                self.writeline(f'fprintf(stderr, "c_{arg_name}: %ld\\n", c_{arg_name}[0]);')

    def dump_arg(self):
        for arg_name, arg_attribute in self.arg_attributes:
            if self.is_out_arg(arg_attribute[0]):
                argv_idx = self.load_args[arg_name]
                argv_sz_idx = self.load_args[arg_name + "_sz"]
                self.writeline(f'if(dump_arg(c_{arg_name}, atoi(argv[{argv_sz_idx}]), argv[{argv_idx}]) == -1){self.open_bracket}')
                with self.code.indent():
                    self.writeline(f'return -1{self.ending}')
                self.writeline(self.closed_bracket)

    def assign_argv_indices(self):
        for arg_name, arg_attribute in self.arg_attributes:
            if arg_name not in self.load_args:
                argv_idx = self.get_argv_idx()
                argv_sz_idx = self.get_argv_idx()
                self.load_args[arg_name] = argv_idx
                self.load_args[arg_name + "_sz"] = argv_sz_idx

    def generate_kernel_declare(self):
        # memref to llvm arguments (memref -> ptr, ptr, i64, <?xi64>, <?xi64>) allocated pointer, aligned pointer, offset, size, stride
        args_type_p = []
        for (_, arg_type) in self.arg_attributes:
            if arg_type[1] in DTYPE_TO_C:
                args_type_p.append(f'{DTYPE_TO_C[arg_type[1]]}*, {DTYPE_TO_C[arg_type[1]]}*, int64_t, int64_t, int64_t')
            elif arg_type[1] == "index":
                args_type_p.append(f'int64_t')

        self.writeline(f"void wrapper_{self.kernel_name}({', '.join(args_type_p)}){self.ending}{self.newline}")

    def generate_args_define(self):
        name_set = set()
        if self.validation:
            self.writeline(f'int *padding = malloc({0x100000*4}ULL){self.ending}') # FIXME. For pooling operation... Some pooling layer use negative offset
        for arg_name, (_, arg_type, _) in self.arg_attributes:
            if arg_name in name_set:
                continue

            if arg_type == "index":
                bits = 64
            elif torch.is_floating_point(torch.tensor([], dtype=arg_type)):
                bits = torch.finfo(arg_type).bits
            elif arg_type == torch.bool:
                bits = 8
            else:
                bits = torch.iinfo(arg_type).bits
            arg_size = f"atoi(argv[{self.load_args[str(arg_name)+'_sz']}])"
            ctype = "int64_t" if arg_type == "index" else DTYPE_TO_C[arg_type]
            self.writeline(f'{ctype}* c_{arg_name} = malloc({arg_size}*{bits // 8}ULL){self.ending}')
            self.writeline(f'fprintf(stderr, "{arg_name} : 0x%lx, size:%d\\n", c_{arg_name}, {arg_size});')
            name_set.add(arg_name)
        self.writeline(self.newline)

    def generate_main(self):
        self.writeline(f'{self.newline}int main(int argc, char *argv[]) {self.open_bracket}{self.newline}')
        with self.code.indent():
            self.assign_argv_indices()
            if self.validation:
                self.generate_args_define()
                self.load_arg()
                self.writeline(self.newline)
            else:
                self.generate_args_define()

            func_arguments = []
            for arg_name, (arg_type, arg_dtype, arg_shape) in self.arg_attributes:
                if MLIRKernelArgs.is_mlir_arg_var(arg_type):
                    func_arguments.append(f"c_{arg_name}[0]")
                else:
                    func_arguments.append(f"c_{arg_name}, c_{arg_name}, 0, atoi(argv[{self.load_args[arg_name+'_sz']}]), 1")
            self.writeline(f"wrapper_{self.kernel_name}({', '.join(func_arguments)}){self.ending}{self.newline}")

            if self.validation:
                self.dump_arg()

            self.write_exit()
        self.writeline(self.closed_bracket)