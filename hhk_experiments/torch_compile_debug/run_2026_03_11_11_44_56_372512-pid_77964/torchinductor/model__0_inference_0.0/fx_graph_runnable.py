
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHSIM_DIR'] = '/workspace/PyTorchSim'
os.environ['PYTORCH_VERSION'] = '2.8.0'
os.environ['TORCHSIM_LLVM_PATH'] = '/riscv-llvm/bin'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_root'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.8.0+cu126
# torch cuda version: 12.6
# torch git version: a1cb3cc05d46d198467bebbb6e8fba50a325d4e7


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        mul = torch.ops.aten.mul.Scalar(convert_element_type, 0.29730177875068026);  convert_element_type = None
        full_default = torch.ops.aten.full.default([1, 32], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        iota = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        le = torch.ops.aten.le.Scalar(sub, 0);  sub = None
        logical_and = torch.ops.aten.logical_and.default(le, full_default);  le = full_default = None
        full_default_1 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        full_default_2 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(logical_and, full_default_2, full_default_1);  logical_and = full_default_2 = full_default_1 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2);  convert_element_type_1 = None
        expand = torch.ops.aten.expand.default(unsqueeze_2, [1, 8, 4, 32, 128]);  unsqueeze_2 = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view = torch.ops.aten.view.default(clone, [1, 32, 32, 128]);  clone = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2);  convert_element_type_2 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_3, [1, 8, 4, 32, 128]);  unsqueeze_3 = None
        clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1 = torch.ops.aten.view.default(clone_1, [1, 32, 32, 128]);  clone_1 = None
        permute = torch.ops.aten.permute.default(view, [0, 1, 3, 2]);  view = None
        mul_1 = torch.ops.aten.mul.Scalar(permute, 0.29730177875068026);  permute = None
        expand_2 = torch.ops.aten.expand.default(mul, [1, 32, 1, 128]);  mul = None
        view_2 = torch.ops.aten.view.default(expand_2, [32, 1, 128]);  expand_2 = None
        expand_3 = torch.ops.aten.expand.default(mul_1, [1, 32, 128, 32]);  mul_1 = None
        view_3 = torch.ops.aten.view.default(expand_3, [32, 128, 32]);  expand_3 = None
        bmm = torch.ops.aten.bmm.default(view_2, view_3);  view_2 = view_3 = None
        view_4 = torch.ops.aten.view.default(bmm, [1, 32, 1, 32]);  bmm = None
        add = torch.ops.aten.add.Tensor(view_4, where);  view_4 = where = None
        amax = torch.ops.aten.amax.default(add, [-1], True)
        sub_1 = torch.ops.aten.sub.Tensor(add, amax);  amax = None
        exp = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        eq = torch.ops.aten.eq.Scalar(add, -inf);  add = None
        logical_not = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1 = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1 = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_3 = torch.ops.aten.full.default([1, 32, 1, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(logical_not_1, full_default_3, div);  logical_not_1 = full_default_3 = div = None
        expand_4 = torch.ops.aten.expand.default(where_1, [1, 32, 1, 32]);  where_1 = None
        view_5 = torch.ops.aten.view.default(expand_4, [32, 1, 32]);  expand_4 = None
        expand_5 = torch.ops.aten.expand.default(view_1, [1, 32, 32, 128]);  view_1 = None
        view_6 = torch.ops.aten.view.default(expand_5, [32, 32, 128]);  expand_5 = None
        bmm_1 = torch.ops.aten.bmm.default(view_5, view_6);  view_5 = view_6 = None
        view_7 = torch.ops.aten.view.default(bmm_1, [1, 32, 1, 128]);  bmm_1 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(view_7, torch.float16);  view_7 = None
        return (convert_element_type_4,)
        
def load_args(reader):
    buf0 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (1, 8, 32, 128), dtype=torch.float16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 65536, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (1, 8, 32, 128), dtype=torch.float16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 8192, device=device(type='npu', index=0), dtype_hint=torch.float16)
    reader.tensor(buf2, (1, 32, 1, 128), dtype=torch.float16, is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)