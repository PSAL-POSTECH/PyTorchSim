class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[1, 8, 32, 128]", arg1_1: "f16[1, 8, 32, 128]", arg2_1: "f16[1, 32, 1, 128]"):
         # File: /workspace/PyTorchSim/hhk_experiments/attention_test.py:118 in forward, code: attn_output = F.scaled_dot_product_attention(
        convert_element_type: "f32[1, 32, 1, 128]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        convert_element_type_1: "f32[1, 8, 32, 128]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        convert_element_type_2: "f32[1, 8, 32, 128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        mul: "f32[1, 32, 1, 128]" = torch.ops.aten.mul.Scalar(convert_element_type, 0.29730177875068026);  convert_element_type = None
        full_default: "b8[1, 32]" = torch.ops.aten.full.default([1, 32], True, dtype = torch.bool, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        iota: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze: "i64[1, 32]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='npu', index=0), requires_grad = False)
        unsqueeze_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub: "i64[1, 32]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        le: "b8[1, 32]" = torch.ops.aten.le.Scalar(sub, 0);  sub = None
        logical_and: "b8[1, 32]" = torch.ops.aten.logical_and.default(le, full_default);  le = full_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where: "f32[1, 32]" = torch.ops.aten.where.self(logical_and, full_default_2, full_default_1);  logical_and = full_default_2 = full_default_1 = None
        unsqueeze_2: "f32[1, 8, 1, 32, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 2);  convert_element_type_1 = None
        expand: "f32[1, 8, 4, 32, 128]" = torch.ops.aten.expand.default(unsqueeze_2, [1, 8, 4, 32, 128]);  unsqueeze_2 = None
        clone: "f32[1, 8, 4, 32, 128]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view: "f32[1, 32, 32, 128]" = torch.ops.aten.view.default(clone, [1, 32, 32, 128]);  clone = None
        unsqueeze_3: "f32[1, 8, 1, 32, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 2);  convert_element_type_2 = None
        expand_1: "f32[1, 8, 4, 32, 128]" = torch.ops.aten.expand.default(unsqueeze_3, [1, 8, 4, 32, 128]);  unsqueeze_3 = None
        clone_1: "f32[1, 8, 4, 32, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_1: "f32[1, 32, 32, 128]" = torch.ops.aten.view.default(clone_1, [1, 32, 32, 128]);  clone_1 = None
        permute: "f32[1, 32, 128, 32]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2]);  view = None
        mul_1: "f32[1, 32, 128, 32]" = torch.ops.aten.mul.Scalar(permute, 0.29730177875068026);  permute = None
        expand_2: "f32[1, 32, 1, 128]" = torch.ops.aten.expand.default(mul, [1, 32, 1, 128]);  mul = None
        view_2: "f32[32, 1, 128]" = torch.ops.aten.view.default(expand_2, [32, 1, 128]);  expand_2 = None
        expand_3: "f32[1, 32, 128, 32]" = torch.ops.aten.expand.default(mul_1, [1, 32, 128, 32]);  mul_1 = None
        view_3: "f32[32, 128, 32]" = torch.ops.aten.view.default(expand_3, [32, 128, 32]);  expand_3 = None
        bmm: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_2, view_3);  view_2 = view_3 = None
        view_4: "f32[1, 32, 1, 32]" = torch.ops.aten.view.default(bmm, [1, 32, 1, 32]);  bmm = None
        add: "f32[1, 32, 1, 32]" = torch.ops.aten.add.Tensor(view_4, where);  view_4 = where = None
        amax: "f32[1, 32, 1, 1]" = torch.ops.aten.amax.default(add, [-1], True)
        sub_1: "f32[1, 32, 1, 32]" = torch.ops.aten.sub.Tensor(add, amax);  amax = None
        exp: "f32[1, 32, 1, 32]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1, 32, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1, 32, 1, 32]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        eq: "b8[1, 32, 1, 32]" = torch.ops.aten.eq.Scalar(add, -inf);  add = None
        logical_not: "b8[1, 32, 1, 32]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[1, 32, 1, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[1, 32, 1, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_3: "f32[1, 32, 1, 32]" = torch.ops.aten.full.default([1, 32, 1, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        where_1: "f32[1, 32, 1, 32]" = torch.ops.aten.where.self(logical_not_1, full_default_3, div);  logical_not_1 = full_default_3 = div = None
        expand_4: "f32[1, 32, 1, 32]" = torch.ops.aten.expand.default(where_1, [1, 32, 1, 32]);  where_1 = None
        view_5: "f32[32, 1, 32]" = torch.ops.aten.view.default(expand_4, [32, 1, 32]);  expand_4 = None
        expand_5: "f32[1, 32, 32, 128]" = torch.ops.aten.expand.default(view_1, [1, 32, 32, 128]);  view_1 = None
        view_6: "f32[32, 32, 128]" = torch.ops.aten.view.default(expand_5, [32, 32, 128]);  expand_5 = None
        bmm_1: "f32[32, 1, 128]" = torch.ops.aten.bmm.default(view_5, view_6);  view_5 = view_6 = None
        view_7: "f32[1, 32, 1, 128]" = torch.ops.aten.view.default(bmm_1, [1, 32, 1, 128]);  bmm_1 = None
        convert_element_type_4: "f16[1, 32, 1, 128]" = torch.ops.prims.convert_element_type.default(view_7, torch.float16);  view_7 = None
        return (convert_element_type_4,)
        