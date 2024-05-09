from typing import List, Optional, cast

from PyTorchSimFrontend.llvm_template import LLVMTemplate
from PyTorchSimFrontend.llvm_template import LLVMTemplateKernel
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode

GEMM_TEMPLATE = r"""
@sram_accum = dso_local global [{{ TILE_M * TILE_N }} x {{ DATA_TYPE }}] zeroinitializer, align 64

define dso_local void @{{ KERNEL_NAME }}(ptr %X, ptr %Y, ptr %W) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv49 = phi i64 [ 0, %entry ], [ %indvars.iv.next50, %for.cond.cleanup3 ]
  %0 = mul nuw nsw i64 %indvars.iv49, {{ K }}
  %add.ptr = getelementptr inbounds {{ DATA_TYPE }}, ptr %X, i64 %0
  %1 = mul nuw nsw i64 %indvars.iv49, {{ N }}
  %add.ptr20 = getelementptr inbounds {{ DATA_TYPE }}, ptr %W, i64 %1
  br label %for.body4

for.cond.cleanup:
  ret void

for.cond.cleanup3:
  %indvars.iv.next50 = add nuw nsw i64 %indvars.iv49, {{ TILE_M }}
  %cmp = icmp ult i64 %indvars.iv49, {{ M - TILE_M }}
  br i1 %cmp, label %for.cond1.preheader, label %for.cond.cleanup

for.body4:
  %indvars.iv47 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next48, %for.cond.cleanup7 ]
  tail call void @llvm.memset.p0.i64(ptr @sram_accum, i8 0, i64 {{ TILE_M * TILE_N * DATA_SIZE }}, i1 false)
  %invariant.gep = getelementptr inbounds {{ DATA_TYPE }}, ptr %Y, i64 %indvars.iv47
  br label %for.body8

for.cond.cleanup7:
  %add.ptr22 = getelementptr inbounds {{ DATA_TYPE }}, ptr %add.ptr20, i64 %indvars.iv47
  {{ kernel.store_matrix(TILE_M, TILE_N, N, DATA_TYPE, DATA_STYPE, "%add.ptr22", "%call18", "W", DATA_SIZE) }}
  %indvars.iv.next48 = add nuw nsw i64 %indvars.iv47, {{ TILE_N }}
  %cmp2 = icmp ult i64 %indvars.iv47, {{ N - TILE_N }}
  br i1 %cmp2, label %for.body4, label %for.cond.cleanup3

for.body8:
  %indvars.iv = phi i64 [ 0, %for.body4 ], [ %indvars.iv.next, %for.body8 ]
  %add.ptr10 = getelementptr inbounds {{ DATA_TYPE }}, ptr %add.ptr, i64 %indvars.iv
  %call = {{ kernel.load_matrix(TILE_M, TILE_K, K, DATA_TYPE, DATA_STYPE, "%add.ptr10", "X", DATA_SIZE)}}
  %2 = mul nuw nsw i64 %indvars.iv, {{ N }}
  %gep = getelementptr inbounds {{ DATA_TYPE }}, ptr %invariant.gep, i64 %2
  %call16 = {{ kernel.load_matrix(TILE_K, TILE_N, N, DATA_TYPE, DATA_STYPE, "%gep", "Y", DATA_SIZE)}}
  %call17 = call <{{ TILE_M * TILE_N }} x {{ DATA_TYPE }}> @llvm.matrix.multiply.v{{ TILE_M*TILE_K }}{{ DATA_STYPE }}.v{{ TILE_K*TILE_N }}{{ DATA_STYPE }}.v{{ TILE_M*TILE_N }}{{ DATA_STYPE }}(<{{ TILE_N * TILE_K}} x {{ DATA_TYPE }}> %call16, <{{ TILE_M * TILE_K}} x {{ DATA_TYPE }}> %call, i32 {{ TILE_M }}, i32 {{ TILE_K }}, i32 {{ TILE_N }})
  %tmp_acc = load <{{ TILE_M * TILE_N }} x {{ DATA_TYPE }}>, ptr @sram_accum, align 64
  %call18 = fadd <{{ TILE_M * TILE_N }} x {{ DATA_TYPE }} > %call17, %tmp_acc
  store <{{ TILE_M * TILE_N }} x {{ DATA_TYPE }}> %call18, ptr @sram_accum, align 64
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, {{ TILE_K }}
  %cmp6 = icmp ult i64 %indvars.iv, {{ K - TILE_K }}
  br i1 %cmp6, label %for.body8, label %for.cond.cleanup7
}
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)
{% if TILE_M == TILE_N %}
declare <{{TILE_M * TILE_K}} x float> @llvm.matrix.column.major.load.v{{ TILE_M * TILE_K }}{{ DATA_STYPE }}.p0{{ DATA_STYPE }}(ptr , i64, i1, i32, i32) #2
{% else %}
declare <{{TILE_M * TILE_K}} x float> @llvm.matrix.column.major.load.v{{ TILE_M * TILE_K }}{{ DATA_STYPE }}.p0{{ DATA_STYPE }}(ptr , i64, i1, i32, i32) #2
declare <{{TILE_N * TILE_K}} x float> @llvm.matrix.column.major.load.v{{ TILE_N * TILE_K }}{{ DATA_STYPE }}.p0{{ DATA_STYPE }}(ptr , i64, i1, i32, i32) #2
{% endif %}
declare <{{TILE_M * TILE_N}} x float> @llvm.matrix.multiply.v{{ TILE_M*TILE_K }}{{ DATA_STYPE }}.v{{ TILE_K*TILE_N }}{{ DATA_STYPE }}.v{{ TILE_M*TILE_N }}{{ DATA_STYPE }}(<{{ TILE_M*TILE_K }} x {{ DATA_TYPE }}>, < {{ TILE_N*TILE_K }} x {{ DATA_TYPE }}>, i32, i32, i32) #1
declare void @llvm.matrix.column.major.store.v{{ TILE_M * TILE_N }}{{ DATA_STYPE }}.p0{{ DATA_STYPE }}(<{{ TILE_M*TILE_N }} x {{ DATA_TYPE }}>, ptr , i64, i1, i32, i32) #3
"""

class LLVMGemmTemplate(LLVMTemplate):
    def __init__(self, input_nodes, layout, input_reorder=None):
        super().__init__("kernel", input_nodes, layout, input_reorder)

    def render(self,
               kernel: LLVMTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               **kwargs):
        if template_buffer_node is not None:
            self.output_node = template_buffer_node
        if epilogue_nodes is not None and len(epilogue_nodes) > 0:
            self.output_node = cast(Buffer, epilogue_nodes[-1])

        X, W = self.input_nodes[0], self.input_nodes[1]
        Y = self.output_node
        Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

        options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            M=X.get_size()[0],
            N=W.get_size()[1],
            K=X.get_size()[1],
            TILE_M=16,
            TILE_N=16,
            TILE_K=16,
            DATA_TYPE="float",
            DATA_STYPE="f32",
            DATA_SIZE=4,
        )
        code = self._template_from_string(GEMM_TEMPLATE).render(**options)
        kernel.add_loop_info([options["M"], options["N"], options["K"]], [options["TILE_M"], options["TILE_N"], options["TILE_K"]])
        kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=self.input_reorder)
        return code