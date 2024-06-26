import os
import subprocess
import shlex

from torch._inductor.utils import IndentedBuffer
from torch._inductor.codegen import cpp
from torch._inductor.codecache import write_atomic

from PyTorchSimFrontend.llvm_common import LLVMKernelArgs

class LLVMKernelCallerCodeGen():
    """
    Generate C that calls the llvm kernel.
    """

    def __init__(self, validation, arg_attributes):
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

    def get_argv_idx(self):
        self.arg_use_count += 1
        return self.arg_use_count-1

    def write_header(self):
        self.writeline('#include <stdio.h>')
        self.writeline('#include <stdlib.h>')
        if self.validation:
            self.writeline("#include <unistd.h>")
            self.writeline('#include <string.h>')
            self.writeline('#include <fcntl.h>')

    def is_in_arg(self, arg_name):
        value = self.arg_attributes[arg_name][0]
        return LLVMKernelArgs.is_llvm_arg_in(value)

    def is_out_arg(self, arg_name):
        value = self.arg_attributes[arg_name][0]
        return LLVMKernelArgs.is_llvm_arg_out(value)

    def load_arg(self):
        for i, arg_name in enumerate(self.arg_attributes.keys()):
            if self.is_in_arg(arg_name):
                self.writeline(f'if(load_arg({arg_name}, sizeof({arg_name}), argv[{self.get_argv_idx()}]) == -1){self.open_bracket}')
                with self.code.indent():
                    self.writeline(f'return -1{self.ending}')
                self.writeline(self.closed_bracket)

    def dump_arg(self):
        for i, arg_name in enumerate(self.arg_attributes.keys()):
            if self.is_out_arg(arg_name):
                self.writeline(f'if(dump_arg({arg_name}, sizeof({arg_name}), argv[{self.get_argv_idx()}]) == -1){self.open_bracket}')
                with self.code.indent():
                    self.writeline(f'return -1{self.ending}')
                self.writeline(self.closed_bracket)

    def write_exit(self):
        self.writeline(f'return 0{self.ending}')

    def generate_kernel_declare(self):
        args_type_p = [f'{cpp.DTYPE_TO_CPP[arg_type[1]]}*' for arg_type in self.arg_attributes.values()]

        self.writeline(f"void {self.kernel_name}({', '.join(args_type_p)}){self.ending}{self.newline}")

    def generate_args_define(self):
        for arg_name, (_, arg_type, arg_shape) in self.arg_attributes.items():
            self.writeline(f'{cpp.DTYPE_TO_CPP[arg_type]} {arg_name}[atoi(argv[{self.get_argv_idx()}])] __attribute__ ((aligned (4096))){self.ending}')
        self.writeline(self.newline)

    def generate_load_dump_fn(self):
        self.writeline(f'{self.newline}int load_arg(void *arg, int size, const char *path) {self.open_bracket}')
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

        self.writeline(f'{self.newline}int dump_arg(void *arg, int size, const char *path) {self.open_bracket}')
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

    def generate_main(self):
        self.writeline(f'{self.newline}int main(int argc, char *argv[]) {self.open_bracket}{self.newline}')
        with self.code.indent():
            self.generate_args_define()
            if self.validation:
                self.load_arg()
                self.writeline(self.newline)

            self.writeline(f"{self.kernel_name}({', '.join(list(self.arg_attributes))}){self.ending}{self.newline}")

            if self.validation:
                self.dump_arg()

            self.write_exit()
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

    def compile_wih_kernel(self, write_path, llvm_name, wrapper_name, binary_name):
        main_path = os.path.join(write_path, self.add_extention(wrapper_name, 'c'))
        main_obj_path = os.path.join(write_path, self.add_extention(wrapper_name, 'o'))
        kernel_path = os.path.join(write_path, self.add_extention(llvm_name, 's'))
        kernel_obj_path = os.path.join(write_path, self.add_extention(llvm_name, 'o'))

        main_compile = f'riscv64-unknown-elf-gcc -march=rv64gcv -c {main_path} -o {main_obj_path}'
        kernel_compile = f'clang -c --target="riscv64" -march=rv64gcv -O2 -nostdlib {kernel_path} -o {kernel_obj_path}'

        target = os.path.join(write_path, binary_name)
        link = f'riscv64-unknown-elf-gcc -march=rv64gcv {main_obj_path} {kernel_obj_path} -o {target} -lm'

        main_compile_cmd = shlex.split(main_compile)
        kernel_compile_cmd = shlex.split(kernel_compile)
        link_cmd = shlex.split(link)

        try:
            subprocess.check_call(main_compile_cmd)
            subprocess.check_call(kernel_compile_cmd)
            subprocess.check_call(link_cmd)
        except subprocess.CalledProcessError as e:
            print("Command failed with exit code", e.returncode)
            print("Error output:", e.output)
            assert(0)
