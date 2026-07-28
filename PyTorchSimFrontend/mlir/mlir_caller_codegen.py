import os
import math
import subprocess
import shlex
import re
import torch
from torch._inductor.utils import IndentedBuffer
from torch._inductor.codecache import write_atomic
from PyTorchSimFrontend.mlir.mlir_common import MLIRKernelArgs, DTYPE_TO_C

class MLIRKernelCallerCodeGen():
    """
    Generate C that calls the llvm kernel.
    """

    def __init__(self, validation, arg_attributes, cycle_sim=False):
        super().__init__()
        self.code = IndentedBuffer()
        self.ending = ";"
        self.open_bracket = "{"
        self.closed_bracket = "}"
        self.newline = "\n"
        self.kernel_name = "kernel"
        self.validation = validation
        self.n_arg = len(arg_attributes)
        self.arg_attributes = arg_attributes
        self.arg_use_count = 1
        self.load_args = {}
        self.kernel_start_addr = ""
        self.kernel_end_addr = ""
        self.cycle_sim = cycle_sim

    def get_argv_idx(self):
        self.arg_use_count += 1
        return self.arg_use_count-1

    def write_header(self):
        self.writeline('#include <stdio.h>')
        self.writeline('#include <stdlib.h>')
        self.writeline("#include <stdint.h>")
        if self.validation:
            self.writeline("#include <unistd.h>")
            self.writeline('#include <string.h>')
            self.writeline('#include <fcntl.h>')
        global_var_header = "gem5_global_var.h" if self.cycle_sim else "global_var.h"
        self.writeline(f"#include \"{global_var_header}\"")

    def is_in_arg(self, value):
        return MLIRKernelArgs.is_mlir_arg_in(value)

    def is_out_arg(self, value):
        return MLIRKernelArgs.is_mlir_arg_out(value)

    def is_inout_arg(self, value):
        return MLIRKernelArgs.is_mlir_arg_inout(value)

    def load_arg(self):
        for arg_name, arg_attribute in self.arg_attributes:
            if self.is_in_arg(arg_attribute[0]):
                argv_idx = self.get_argv_idx() if arg_name not in self.load_args else self.load_args[arg_name]
                self.load_args[arg_name] = argv_idx
                ctype = DTYPE_TO_C[arg_attribute[1]]
                elem_count = arg_attribute[2]
                size_expr = f'({elem_count}ULL * sizeof({ctype}))'

                self.writeline(f'if(load_arg(c_{arg_name}, {size_expr}, argv[{argv_idx}]) == -1){self.open_bracket}')
                with self.code.indent():
                    self.writeline(f'return -1{self.ending}')
                self.writeline(self.closed_bracket)

    def dump_arg(self):
        for arg_name, arg_attribute in self.arg_attributes:
            if self.is_out_arg(arg_attribute[0]):
                argv_idx = self.get_argv_idx() if not self.is_inout_arg(arg_attribute[0]) else self.load_args[arg_name]
                ctype = DTYPE_TO_C[arg_attribute[1]]
                elem_count = arg_attribute[2]
                size_expr = f'({elem_count}ULL * sizeof({ctype}))'
                self.writeline(f'if(dump_arg(c_{arg_name}, {size_expr}, argv[{argv_idx}]) == -1){self.open_bracket}')
                with self.code.indent():
                    self.writeline(f'return -1{self.ending}')
                self.writeline(self.closed_bracket)

    def write_exit(self):
        self.writeline(f'return 0{self.ending}')

    def generate_kernel_declare(self):
        # memref to llvm arguments (memref -> ptr, ptr, i64, <?xi64>, <?xi64>) allocated pointer, aligned pointer, offset, size, stride
        args_type_p = [f'{DTYPE_TO_C[arg_type[1]]}*, {DTYPE_TO_C[arg_type[1]]}*, int64_t, int64_t, int64_t' for (_, arg_type) in self.arg_attributes]

        self.writeline(f"void wrapper_{self.kernel_name}({', '.join(args_type_p)}){self.ending}{self.newline}")

    def generate_args_define(self):
        name_set = set()
        if self.validation:
            self.writeline(f"int* padding = malloc(0x100000ULL * sizeof(int)){self.ending}")
        for arg_name, (_, arg_type, arg_size, arg_sizes, arg_stride) in self.arg_attributes:
            if not arg_name in name_set:
                if torch.is_floating_point(torch.tensor([], dtype=arg_type)):
                    bits = torch.finfo(arg_type).bits
                elif arg_type == torch.bool:
                    bits = 8
                else:
                    bits = torch.iinfo(arg_type).bits
                buffer_size = int(math.ceil(arg_size * bits // 8 / 64) * 64) * 2 # Round up to 64 bytes + Add some padding for safety
                self.writeline(f'{DTYPE_TO_C[arg_type]}* c_{arg_name} = malloc({buffer_size}ULL){self.ending}')
                name_set.add(arg_name)
        self.writeline(self.newline)

    def generate_main(self):
        self.writeline(f'{self.newline}int main(int argc, char *argv[]) {self.open_bracket}{self.newline}')
        with self.code.indent():
            if self.validation:
                self.generate_args_define()
                self.load_arg()
                self.writeline(self.newline)
            else:
                self.generate_args_define()

            func_arguments = [f"c_{arg_name}, c_{arg_name}, 0, {arg_shape}, 1" for arg_name, (_, arg_type, arg_shape, _, _) in self.arg_attributes]
            self.writeline(f"wrapper_{self.kernel_name}({', '.join(func_arguments)}){self.ending}{self.newline}")

            if self.validation:
                self.dump_arg()

            self.write_exit()
        self.writeline(self.closed_bracket)

    def generate_load_dump_fn(self):
        self.writeline(f'{self.newline}int load_arg(void *arg, size_t size, const char *path) {self.open_bracket}')
        with self.code.indent():
            self.writeline(f'int fd = open(path, 0x00000000){self.ending}')
            self.writeline(f'if (fd == -1) {self.open_bracket}')
            with self.code.indent():
                self.writeline(f'return -1{self.ending}')
            self.writeline(self.closed_bracket)

            self.writeline(f'if (read(fd, arg, size) == -1) {self.open_bracket}')
            with self.code.indent():
                self.writeline(f'return -1{self.ending}')
            self.writeline(self.closed_bracket)
            self.writeline(f'close(fd){self.ending}')
            self.writeline(f'return 0{self.ending}')
        self.writeline(self.closed_bracket)

        self.writeline(f'{self.newline}int dump_arg(void *arg, size_t size, const char *path) {self.open_bracket}')
        with self.code.indent():
            self.writeline(f'int fd = open(path, 0x00000001 | 0x00000040, 0644){self.ending}')
            self.writeline(f'if (fd == -1) {self.open_bracket}')
            with self.code.indent():
                self.writeline(f'return -1{self.ending}')
            self.writeline(self.closed_bracket)

            self.writeline(f'if (write(fd, arg, size) == -1) {self.open_bracket}')
            with self.code.indent():
                self.writeline(f'return -1{self.ending}')
            self.writeline(self.closed_bracket)
            self.writeline(f'close(fd){self.ending}')
            self.writeline(f'return 0{self.ending}')
        self.writeline(self.closed_bracket)


    def writeline(self, line):
        self.code.writeline(line)

    def generate_wrapper_file(self, path, name):
        self.dump_path = path

        self.write_header()
        self.generate_kernel_declare()

        if self.validation:
            self.generate_load_dump_fn()
        self.generate_main()

        write_path = os.path.join(path, name+".c",)
        write_atomic(write_path, self.code.getvalue())
        return

    def add_extention(self, name, extension):
        return name + "." + extension

    def compile_wih_kernel(self, write_path, llvm_name, wrapper_name, binary_name, link_option=""):
        main_path = os.path.join(write_path, self.add_extention(wrapper_name, 'c'))
        main_obj_path = os.path.join(write_path, self.add_extention(wrapper_name, 'o'))
        kernel_obj_path = os.path.join(write_path, self.add_extention(llvm_name, 'o'))

        main_compile = f'riscv64-unknown-elf-gcc -march=rv64gcv -c {main_path} -o {main_obj_path}'

        target = os.path.join(write_path, binary_name)
        link = f'riscv64-unknown-elf-gcc -march=rv64gcv {main_obj_path} {kernel_obj_path} -o {target} -lm {link_option}'

        main_compile_cmd = shlex.split(main_compile)
        link_cmd = shlex.split(link)

        try:
            subprocess.check_call(main_compile_cmd)
            subprocess.check_call(link_cmd)
        except subprocess.CalledProcessError as e:
            print("Command failed with exit code", e.returncode)
            print("Error output:", e.output)
            assert(0)

    def parse_stack_sizes(self, file_path, vlenb=256):
        with open(file_path, 'r') as f:
            stack_sizes_data = f.readlines()

        in_proc = False
        stack_base = None
        dynamic_expr = None
        max_offset = 0

        for line in stack_sizes_data:
            line = line.strip()
            if line.startswith(".cfi_startproc"):
                in_proc = True
                continue
            elif line.startswith(".cfi_endproc") and in_proc:
                if dynamic_expr:
                    total_stack = eval(dynamic_expr, {"vlenb": vlenb})
                    return total_stack
                elif stack_base:
                    return stack_base
                else:
                    return max_offset

            # Skip outer function
            if not in_proc:
                continue

            if line.startswith(".cfi_def_cfa_offset"):
                stack_base = int(line.split()[-1])

            if ".cfi_escape" in line and "#" in line:
                comment = line.split("#")[-1].strip()
                m = re.search(r"sp \+ (\d+)\s*\+\s*(\d+)\s*\*\s*vlenb", comment)
                if m:
                    base, scale = int(m.group(1)), int(m.group(2))
                    dynamic_expr = f"{base} + {scale} * vlenb"

    def get_spad_size(self, binary_path):
        cmd = ["riscv64-unknown-elf-readelf", "-s", binary_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Readelf error: {result.stderr}")

        output = result.stdout
        spad_start = None
        spad_end = None
        for line in output.splitlines():
            if '.spad' in line and 'SECTION' in line:
                parts = line.split()
                spad_start = int(parts[1], 16)
            elif 'spad_end' in line:
                parts = line.split()
                spad_end = int(parts[1], 16)

        if spad_start is None or spad_end is None:
            return 0
        spad_size = spad_end - spad_start
        return spad_size