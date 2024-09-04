from typing import List, Optional, Sequence

import torch
from torch._inductor.lowering import lowerings
from torch._inductor.kernel.mm_common import mm_args
from torch._inductor import ir
from torch._inductor.virtualized import V
from torch._inductor.ir import TensorBox
from PyTorchSimFrontend.mlir.mlir_gemm_template import MLIRGemmTemplate

aten = torch.ops.aten

def tuned_mm(mat1, mat2, * ,layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    mlir_template = MLIRGemmTemplate([mat1, mat2], layout)

    return mlir_template.generate().output_node()

lowerings.update({getattr(aten.mm, overload): tuned_mm for overload in aten.mm.overloads()})