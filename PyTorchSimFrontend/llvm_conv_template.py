import os, math
from typing import List, Optional, cast
from PyTorchSimFrontend.llvm_common import LLVMKernelArgs
from PyTorchSimFrontend.llvm_template import LLVMTemplate
from PyTorchSimFrontend.llvm_template import LLVMTemplateKernel
from torch._inductor.ir import Buffer
from torch._inductor.ir import IRNode
from torch._inductor.codecache import get_hash

CONV2D_TEMPLATE = r"""
@sram_accum = dso_local global [{{ TILE_M * TILE_N }} x {{ DATA_TYPE }}] zeroinitializer, align 64

define dso_local void @{{ KERNEL_NAME }}(ptr %X, ptr %Y, ptr %B, ptr %W) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv59 = phi i64 [ 0, %entry ], [ %indvars.iv.next60, %for.cond.cleanup3 ]
  %0 = mul nuw nsw i64 %indvars.iv59, {{ N }}
  %add.ptr = getelementptr inbounds {{ DATA_TYPE }}, ptr %B, i64 %0
  %1 = mul nuw nsw i64 %indvars.iv59, {{ K }}
  %add.ptr13 = getelementptr inbounds {{ DATA_TYPE }}, ptr %X, i64 %1
  %add.ptr27 = getelementptr inbounds {{ DATA_TYPE }}, ptr %W, i64 %0
  br label %for.body4

for.cond.cleanup:
  ret void

for.cond.cleanup3:
  %indvars.iv.next60 = add nuw nsw i64 %indvars.iv59, {{ TILE_M }}
  %cmp = icmp ult i64 %indvars.iv59, {{ M - TILE_M }}
  br i1 %cmp, label %for.cond1.preheader, label %for.cond.cleanup

for.body4:
  %indvars.iv57 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next58, %for.cond.cleanup9 ]
  %add.ptr6 = getelementptr inbounds {{ DATA_TYPE }}, ptr %add.ptr, i64 %indvars.iv57
  %call = {{ kernel.load_matrix(TILE_M, TILE_N, N, DATA_TYPE, DATA_STYPE, "%add.ptr6", "B", DATA_SIZE)}}
  tail call void @llvm.memset.p0.i64(ptr @sram_accum, i8 0, i64 {{ TILE_M * TILE_N * DATA_SIZE }}, i1 false)
  %invariant.gep = getelementptr inbounds {{ DATA_TYPE }}, ptr %Y, i64 %indvars.iv57
  br label %for.body10

for.cond.cleanup9:
  %call24 = fadd <{{ TILE_M * TILE_N }} x {{ DATA_TYPE }} > %call, %call18
  %add.ptr29 = getelementptr inbounds {{ DATA_TYPE }}, ptr %add.ptr27, i64 %indvars.iv57
  {{ kernel.store_matrix(TILE_M, TILE_N, N, DATA_TYPE, DATA_STYPE, "%add.ptr29", "%call24", "W", DATA_SIZE) }}
  %indvars.iv.next58 = add nuw nsw i64 %indvars.iv57, {{ TILE_N }}
  %cmp2 = icmp ult i64 %indvars.iv57, {{ N - TILE_N }}
  br i1 %cmp2, label %for.body4, label %for.cond.cleanup3

for.body10:
  %indvars.iv = phi i64 [ 0, %for.body4 ], [ %indvars.iv.next, %for.body10 ]
  %add.ptr15 = getelementptr inbounds {{ DATA_TYPE }}, ptr %add.ptr13, i64 %indvars.iv
  %call16 = {{ kernel.load_matrix(TILE_M, TILE_K, K, DATA_TYPE, DATA_STYPE, "%add.ptr15", "X", DATA_SIZE)}}
  %2 = mul nuw nsw i64 %indvars.iv, {{ N }}
  %gep = getelementptr inbounds {{ DATA_TYPE }}, ptr %invariant.gep, i64 %2
  %call22 = {{ kernel.load_matrix(TILE_K, TILE_N, N, DATA_TYPE, DATA_STYPE, "%gep", "Y", DATA_SIZE)}}
  %call23 = call <{{ TILE_M * TILE_N }} x {{ DATA_TYPE }}> @llvm.matrix.multiply.v{{ TILE_M*TILE_K }}{{ DATA_STYPE }}.v{{ TILE_K*TILE_N }}{{ DATA_STYPE }}.v{{ TILE_M*TILE_N }}{{ DATA_STYPE }}(<{{ TILE_N * TILE_K}} x {{ DATA_TYPE }}> %call22, <{{ TILE_M * TILE_K}} x {{ DATA_TYPE }}> %call16, i32 {{ TILE_M }}, i32 {{ TILE_K }}, i32 {{ TILE_N }})

  %tmp_acc = load <{{ TILE_M * TILE_N }} x {{ DATA_TYPE }}>, ptr @sram_accum, align 64
  %call18 = fadd <{{ TILE_M * TILE_N }} x {{ DATA_TYPE }} > %call23, %tmp_acc
  store <{{ TILE_M * TILE_N }} x {{ DATA_TYPE }}> %call18, ptr @sram_accum, align 64

  %indvars.iv.next = add nuw nsw i64 %indvars.iv, {{ TILE_K }}
  %cmp8 = icmp ult i64 %indvars.iv, {{ K - TILE_K }}
  br i1 %cmp8, label %for.body10, label %for.cond.cleanup9
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

CONV2D_FUNC = r"""
def {{ FUNC_NAME }}({{ INPUT }}, {{ WEIGHT }}, {{ BIAS }}, {{ OUT }}):
  {{ INPUT }}_cpu = {{ INPUT }}.cpu()
  {{ WEIGHT }}_cpu = {{ WEIGHT }}.cpu()
  {{ BIAS }}_cpu = {{ BIAS }}.cpu()
  {{ OUT }}_cpu = {{ OUT }}.cpu()

  # Torch support NCHW, so we need to transpose for now
  {{ INPUT }}_cpu = {{ INPUT }}_cpu.permute(0, 2, 3, 1)
  {{ WEIGHT }}_cpu = {{ WEIGHT }}_cpu.permute(0, 2, 3, 1)
  {{ OUT }}_cpu = {{ OUT }}_cpu.permute(0, 2, 3, 1)
  {{ OUT }}_cpu.zero_()

  input_shape = {{ INPUT }}_cpu.shape
  weight_shape = {{ WEIGHT }}_cpu.shape
  output_shape = {{ OUT }}_cpu.shape

  input_pad_shape = (input_shape[0], input_shape[1]+2*{{ PADDING_H }}, input_shape[2]+2*{{ PADDING_W }}, input_shape[3])
  input_pad = torch.zeros(input_pad_shape)

  if {{ PADDING_H }} != 0 and {{ PADDING_W }} != 0:
      input_pad[:, {{ PADDING_H }}:-{{ PADDING_H }}, {{ PADDING_W }}:-{{ PADDING_W }}, :] = {{ INPUT }}_cpu
  elif {{ PADDING_H }} != 0:
      input_pad[:, {{ PADDING_H }}:-{{ PADDING_H }}, :, :] = {{ INPUT }}_cpu
  elif {{ PADDING_W }} != 0:
      input_pad[:,:, {{ PADDING_W }}:-{{ PADDING_W }}, :] = {{ INPUT }}_cpu
  else:
      input_pad = {{ INPUT }}_cpu

  {% if VALIDATION_MODE %}
  {% endif %}

  for kh in range(weight_shape[1]):
      for kw in range(weight_shape[2]):
          input_tile = input_pad[:, kh:input_pad_shape[1]-(weight_shape[1]-1)+kh, kw:input_pad_shape[2]-(weight_shape[2]-1)+kw, :]
          input_tile = input_tile[:,::{{ STRIDE_H }},::{{ STRIDE_W }}, :]
          kernel_tile = {{ WEIGHT }}_cpu[:, kh, kw, :].t()
          input_tile = input_tile.reshape(-1, input_pad_shape[3])
          {% if VALIDATION_MODE %}
          if kh == 0 and kw == 0:
              {{ KERNEL_NAME }}(input_tile, kernel_tile, {{ OUT }}_cpu, {{ OUT }}_cpu, intermediate_op=0b01)
          elif kh == weight_shape[1]-1 and kw == weight_shape[2]-1:
              {{ KERNEL_NAME }}(input_tile, kernel_tile, {{ OUT }}_cpu, {{ OUT }}_cpu, intermediate_op=0b10)
          else:
              {{ KERNEL_NAME }}(input_tile, kernel_tile, {{ OUT }}_cpu, {{ OUT }}_cpu, intermediate_op=0b11)
          {% else %}
          {{ KERNEL_NAME }}(input_tile, kernel_tile, {{ OUT }}_cpu, {{ OUT }}_cpu)  # input, weight, bias, out
          {% endif %}

  {{ OUT }}_cpu = {{ OUT }}_cpu.permute(0, 3, 1, 2)
  {{ OUT }}.copy_({{ OUT }}_cpu)
"""

class LLVMConvTemplate(LLVMTemplate):
    def __init__(self, input_nodes, layout, input_reorder=None, **kwargs):
      super().__init__("kernel", input_nodes, layout, input_reorder)
      self.stride = kwargs["stride"]
      self.padding = kwargs["padding"]
      self.dilation = kwargs["dilation"]
      weight_shape = [str(i) for i in input_nodes[1].layout.size]
      self.function_name = "Conv2D_" + "_".join(weight_shape)
      self.gemm_args = ['input', 'weight', 'bias', 'output']

    def render(self,
               kernel: LLVMTemplateKernel,
               template_buffer_node = None,
               epilogue_nodes: Optional[List[IRNode]] = None,
               **kwargs):
      if template_buffer_node is not None:
        self.output_node = template_buffer_node
      if epilogue_nodes is not None and len(epilogue_nodes) > 0:
        self.output_nodes = cast(Buffer, epilogue_nodes[-1])

      X, W = self.input_nodes[0], self.input_nodes[1]
      Y = self.output_node
      Bias = None if len(self.input_nodes) == 2 else self.input_nodes[2]

      input_h = X.get_size()[2]
      input_w = X.get_size()[3]
      kernel_h = W.get_size()[2]
      kernel_w = W.get_size()[3]
      i_c = X.get_size()[1]
      o_c = W.get_size()[0]

      weight_shape = self.input_nodes[1].get_size()
      i_tile_h = int((input_h + 2*self.padding[0] - (weight_shape[2]-1) - 1) / self.stride[0]) + 1
      i_tile_w = int((input_w + 2*self.padding[1] - (weight_shape[3]-1) - 1) / self.stride[1]) + 1

      gemm_m = i_tile_h * i_tile_w
      gemm_n = o_c
      gemm_k = i_c

      options = dict(
          KERNEL_NAME=self.name,
          kernel=kernel,
          M=gemm_m,
          N=gemm_n,
          K=gemm_k,
          TILE_M=4,
          TILE_N=4,
          TILE_K=4,
          DATA_TYPE="float",
          DATA_STYPE="f32",
          DATA_SIZE=4,
      )
      code = self._template_from_string(CONV2D_TEMPLATE).render(**options)
      kernel.add_loop_info([options["M"], options["N"], options["K"]], [options["TILE_M"], options["TILE_N"], options["TILE_K"]])
      kernel.def_kernel(inputs=[X, W, Bias], outputs=[Y], names_str="X, W, Bias, Y", input_reorder=self.input_reorder)

      self.hash_value = get_hash(code.strip())
      return code

    def function_render(self, kernel_name, input_args):

      options = dict(
        KERNEL_NAME=kernel_name,
        FUNC_NAME=self.function_name,
        INPUT=input_args[0],
        WEIGHT=input_args[1],
        BIAS=input_args[2],
        OUT=input_args[3],
        STRIDE_H=self.stride[0],
        STRIDE_W=self.stride[1],
        PADDING_H=self.padding[0],
        PADDING_W=self.padding[1],
        DILATION_H=self.dilation[0],
        DILATION_W=self.dilation[1],
        VALIDATION_MODE=int(os.environ.get('TORCH_VALIDATION_MODE', default="True") == "True"),
        HASH_VALUE=self.hash_value
      )
      code = self._template_from_string(CONV2D_FUNC).render(**options)
      return code

    def get_arg_attributes(self):
      arg_attributes = {}

      input_shape = self.input_nodes[0].get_size()
      weight_shape = self.input_nodes[1].get_size()
      gemm_h = int((input_shape[2] + 2*self.padding[0] - (weight_shape[2]-1) - 1) / self.stride[0]) + 1
      gemm_w = int((input_shape[3] + 2*self.padding[1] - (weight_shape[3]-1) - 1) / self.stride[1]) + 1

      gemm_input_shape = [input_shape[0],input_shape[1],gemm_h, gemm_w]
      gemm_weight_shape = [weight_shape[0],weight_shape[1],1,1]
      gemm_output_shape = [gemm_input_shape[2]*gemm_input_shape[3], gemm_weight_shape[0]] # Consider Batch size 1

      arg_attributes[self.gemm_args[0]] = [LLVMKernelArgs.LLVM_ARGS_IN, self.input_nodes[0].layout.dtype, math.prod(gemm_input_shape)]
      arg_attributes[self.gemm_args[1]] = [LLVMKernelArgs.LLVM_ARGS_IN, self.input_nodes[1].layout.dtype, math.prod(gemm_weight_shape)]
      arg_attributes[self.gemm_args[2]] = [LLVMKernelArgs.LLVM_ARGS_IN, self.input_nodes[0].layout.dtype, math.prod(gemm_output_shape)]
      arg_attributes[self.gemm_args[3]] = [LLVMKernelArgs.LLVM_ARGS_OUT, self.input_nodes[0].layout.dtype, math.prod(gemm_output_shape)]

      return arg_attributes