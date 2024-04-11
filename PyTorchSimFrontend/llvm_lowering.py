import torch
from torch._inductor.lowering import lowerings
from torch._inductor.kernel.mm_common import mm_args
from PyTorchSimFrontend.llvm_gemm_template import LLVMGemmTemplate

aten = torch.ops.aten

def tuned_mm(mat1, mat2, * ,layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    llvm_template = LLVMGemmTemplate([mat1, mat2], layout)

    return llvm_template.generate().output_node()

lowerings.update({getattr(aten.mm, overload): tuned_mm for overload in aten.mm.overloads()})