# mlir_lowering.py
# 이 파일은 PyTorch Inductor의 lowering 단계에서 특정 aten 연산들을 MLIR 템플릿 또는 커스텀 구현으로 매핑합니다.
# 각 lowering 함수는 입력 IR(TensorBox)들을 받아 MLIR 템플릿을 생성하거나, 외부 커널 호출을 준비합니다.

from typing import List, Optional, Sequence  

import torch  
from torch._inductor.lowering import lowerings, index_impl 
from torch._inductor.kernel.mm_common import mm_args 
# from torch._inductor.select_algorithm import ExternKernelChoice  
from torch._inductor import ir  
from torch._inductor.virtualized import V  
from torch._inductor.ir import TensorBox  
from PyTorchSimFrontend.extension_op import MLIRExternKernelChoice  
from PyTorchSimFrontend.mlir.mlir_gemm_template import MLIRGemmTemplate  
from PyTorchSimFrontend.mlir.mlir_bmm_template import MLIRBMMTemplate  
from PyTorchSimFrontend.mlir.mlir_conv_template import MLIRConvTemplate  
from PyTorchSimFrontend.mlir.mlir_conv_mt_template import MLIRConvMultiTileTemplate  
from PyTorchSimFrontend.mlir.mlir_conv_sb_template import MLIRConvSingleBatchTemplate  
from PyTorchSimFrontend.mlir.mlir_conv_sbs_template import MLIRConvSingleBatchStridedTemplate 
from PyTorchSimFrontend.mlir.mlir_maxpool_template import MLIRMaxPoolTemplate  
from PyTorchSimFrontend.mlir.mlir_foobar_template import MLIRFoobarTemplate
from PyTorchSimFrontend import extension_config 

# shortcut: aten ops에 접근하기 쉽게 변수에 할당합니다.
aten = torch.ops.aten
# sparse mm 연산을 MLIR 외부 커널로 매핑하기 위한 래퍼를 생성합니다. (외부 커널 이름은 "custom_op::sparse_addmm")
aten_spmm = MLIRExternKernelChoice(torch.sparse.mm, "custom_op::sparse_addmm")

#tuned_mm, tuned_addmm, tuned_bmm는 각각 mm, addmm, bmm 연산을 MLIR 템플릿으로 변환하는 함수입니다.

def tuned_mm(mat1, mat2, * ,layout=None):
    # mm (행렬 곱) 연산을 받아 MLIR GEMM 템플릿으로 변환합니다.
    # mm_args는 입력들의 형상/레이아웃 정보를 통일하고 (m,n,k,layout,mat1,mat2)를 반환합니다.
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    # GEMM 템플릿을 생성합니다. 레이아웃 정보는 템플릿 생성에 필요합니다.
    mlir_template = MLIRGemmTemplate([mat1, mat2], layout)

    # 템플릿을 생성하고 출력 IR 노드를 반환합니다. generate()는 템플릿의 IR 그래프를 구성합니다.
    return mlir_template.generate(input_nodes=[mat1, mat2], layout=layout).output_node()


def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    # addmm (alpha * mat1 @ mat2 + beta * inp) 연산을 MLIR GEMM 템플릿으로 변환합니다.
    # mm_args는 입력/출력 크기에 맞춰 inp를 확장하거나 필요한 정보를 반환합니다.
    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(mat1, mat2, inp, layout=layout)
    # GEMM 템플릿에 inp(바이어스/누적 결과)를 포함하여 생성합니다.
    mlir_template = MLIRGemmTemplate([mat1, mat2, inp_expanded], layout)

    # 생성된 템플릿의 출력 노드를 반환합니다. addmm의 결과를 IR상 노드로 대체합니다.
    return mlir_template.generate().output_node()


def tuned_bmm(mat1, mat2, *, layout=None):
    # 배치 행렬곱(bmm)을 BMM 템플릿으로 변환합니다.
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    mlir_template = MLIRBMMTemplate([mat1, mat2], layout)

    # 템플릿을 생성 후 출력 노드를 반환합니다.
    return mlir_template.generate().output_node()


def conv_layout(
    x: TensorBox,
    weight: TensorBox,
    bias: Optional[TensorBox],
    stride: Sequence[int],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    transposed: bool,
    output_padding: tuple[int, ...],
    groups: int,
) -> ir.Layout:
    """Determine output layout for a convolution

    이 함수는 가상 실행(fake mode)으로 convolution 연산의 출력 텐서의 shape/stride를 구하고
    그 결과로 Inductor의 FixedLayout을 반환합니다. 템플릿을 만들 때 출력 레이아웃을 알아야 하므로 필요합니다.
    """
    # V.graph.fake_mode를 사용하면 실제 데이터를 계산하지 않고 크기 추론을 할 수 있습니다.
    with V.graph.fake_mode:
        # aten.convolution을 호출하여 출력의 사이즈와 스트라이드를 얻습니다.
        output = torch.ops.aten.convolution(
            ir.ir_node_to_tensor(x, guard_shape=True),  # TensorBox를 텐서 형태로 변환하여 크기 추론에 사용
            ir.ir_node_to_tensor(weight, guard_shape=True),
            ir.ir_node_to_tensor(bias, guard_shape=True),
            stride,
            tuple(V.graph.sizevars.size_hint(p) for p in padding),  # padding 값은 sizevars를 통해 힌트를 사용
            dilation,
            transposed,
            tuple(V.graph.sizevars.size_hint(p) for p in output_padding),
            groups,
        )
        # 출력의 사이즈/스트라이드를 Inductor 내부 포맷으로 변환합니다.
        sizes = ir.convert_shape_to_inductor(output.size())
        stride = ir.convert_shape_to_inductor(output.stride())

    # FixedLayout은 장치, dtype, 크기, 스트라이드를 고정된 레이아웃으로 표현합니다.
    return ir.FixedLayout(
        x.get_device(),
        x.get_dtype(),
        sizes,
        stride,
    )


def convolution(
    x: TensorBox,
    weight: TensorBox,
    bias: TensorBox,
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    transposed: bool,
    output_padding: List[int],
    groups: int,
):
    # 입력으로 들어오는 리스트들을 튜플로 바꿔 불변성을 보장하고 일관된 타입으로 사용합니다.
    stride = tuple(stride)
    padding = tuple(padding)
    dilation = tuple(dilation)
    output_padding = tuple(output_padding)

    # 템플릿 생성 시 필요한 인자들을 kwargs로 모아둡니다.
    kwargs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "transposed": transposed,
        "output_padding": output_padding,
        "groups": groups,
    }

    # TensorBox는 지연(lazy) 표현일 수 있기 때문에 실제 값을 생성해야 템플릿 생성에 사용할 수 있습니다.
    x.realize()
    weight.realize()
    # 컨볼루션 템플릿은 채널-마지막 레이아웃을 기대할 때가 있어 보장하기 위한 헬퍼입니다.
    x = ir.ExternKernel.require_channels_last(x)
    # 배치 및 입력 채널 수를 빠르게 확인하여 템플릿 분기를 결정합니다.
    BATCH = x.layout.size[0]
    I_C = x.layout.size[1]
    weight = ir.ExternKernel.require_channels_last(weight)
    # 출력 레이아웃을 미리 계산하여 템플릿에 넘깁니다.
    layout = conv_layout(x, weight, None, **kwargs)

    # 적절한 컨볼루션 템플릿을 선택합니다. 싱글 배치, 스트라이드 여부, 멀티 타일 등 상황에 따라 분기합니다.
    if BATCH == 1 and stride[0] == 1 and extension_config.CONFIG_SINGLE_BATCH_CONV:
        # 배치=1 & stride=1용 간소화된 구현 사용
        mlir_template = MLIRConvSingleBatchTemplate([x, weight, bias], layout, **kwargs)
    elif BATCH == 1 and stride[0] != 1 and extension_config.CONFIG_SINGLE_BATCH_CONV:
        # 배치=1 & stride!=1인 경우 다른 템플릿 사용
        mlir_template = MLIRConvSingleBatchStridedTemplate([x, weight, bias], layout, **kwargs)
    elif I_C < extension_config.vpu_num_lanes // 8 and extension_config.CONFIG_MULTI_TILE_CONV: # 8 is hard-coded for now. This should be changed to a better heuristic.
        # 입력 채널이 작아 멀티-타일 전략이 효과적일 때 multi-tile 템플릿 사용
        mlir_template = MLIRConvMultiTileTemplate([x, weight, bias], layout, **kwargs)
    else:
        # 기본 일반 컨볼루션 템플릿 사용
        mlir_template = MLIRConvTemplate([x, weight, bias], layout, **kwargs)
    # 생성된 템플릿의 출력 노드를 반환합니다.
    return mlir_template.generate().output_node()


def maxpool_layout(
    x: TensorBox,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    ceil_mode: bool,
) -> ir.Layout:
    """Determine output layout for a maxpool

    conv_layout와 유사하게 maxpool 연산의 출력 layout을 추론합니다.
    """
    # fake_mode로 실제 데이터 없이 출력 크기를 추론합니다.
    with V.graph.fake_mode:
        output, _ = torch.ops.aten.max_pool2d_with_indices(
            ir.ir_node_to_tensor(x, guard_shape=True),
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        )
        # 출력의 사이즈와 스트라이드를 Inductor 형식으로 변환
        sizes = ir.convert_shape_to_inductor(output.size())
        stride = ir.convert_shape_to_inductor(output.stride())

    # FixedLayout으로 반환
    return ir.FixedLayout(
        x.get_device(),
        x.get_dtype(),
        sizes,
        stride,
    )


def custom_maxpool(
    x: TensorBox,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int] = [1, 1],
    ceil_mode: bool = False
):
    # maxpool 호출을 템플릿화하여 MLIRMaxPoolTemplate로 처리합니다.
    kwargs = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
    }
    # 출력 레이아웃을 미리 계산
    layout = maxpool_layout(x, kernel_size, stride, padding, dilation, ceil_mode)
    mlir_template = MLIRMaxPoolTemplate([x], layout, **kwargs)
    # TensorBox 실체화
    x.realize()
    # 템플릿을 생성하여 출력 노드를 반환합니다.
    template_node = mlir_template.generate().output_node()
    # indices(인덱스)는 현재 사용하지 않으므로 dummy x와 함께 반환합니다. (FIXME: indices 처리 필요)
    return template_node, x # FIXME: x is dummy IRNode, indices are not used in our case


def sparse_addmm(*args, **kwargs):
    # 희소 행렬 연산의 예시적 외부 커널 매핑
    _, sp_mat1, sp_mat2 = args  # 첫 인자는 out placeholder, 그 다음이 두 희소 행렬
    mat1_layout = sp_mat1.layout  # 희소 행렬의 레이아웃을 참고
    # out의 range와 dims 정보를 사용해 출력 크기를 계산합니다. (구조체 접근은 API 의존적)
    out_range = args[0].data.data.data.ranges
    size = [out_range[i] for i in args[0].data.dims]
    # FlexibleLayout을 만들어서 외부 커널에 필요한 레이아웃 정보를 제공합니다.
    layout = ir.FlexibleLayout(
            device=mat1_layout.device, dtype=mat1_layout.dtype, size=size  # FIXME: Example code for aten op overwrite by externkernel call
        )
    # 외부 스파스 행렬 연산으로 바인딩하고 출력 노드를 반환합니다.
    return aten_spmm.bind((sp_mat1, sp_mat2), layout).output_node()


def custom_unsafe_index(x, indices):
    # 안전하지 않은 인덱스 접근은 간단히 인덱스 구현으로 처리하되,
    # TensorBox인 경우 실체화(realize)하여 실제 텐서/데이터가 준비되도록 합니다.
    # 주석: 간접 접근(indirect access) + indexed_expression + computation은 fusion할 수 없습니다.
    if isinstance(x, TensorBox):
        x.realize()
    # index_impl을 호출할 때 check=False로 하여 일부 검사를 건너뜁니다(unsafe 행동을 허용).
    return index_impl(x, indices, check=False)

def custom_foobar(a, *args, **kwargs):
    a.realize()
    layout = a.layout
    mlir_template = MLIRFoobarTemplate([a], layout)
    return mlir_template.generate().output_node()


# aten 연산 오버로드들을 위에서 정의한 커스텀 lowering 함수로 등록합니다.
lowerings.update({getattr(aten.mm, overload): tuned_mm for overload in aten.mm.overloads()})
lowerings.update({getattr(aten.addmm, overload): tuned_addmm for overload in aten.addmm.overloads()})
lowerings.update({getattr(aten.convolution, overload): convolution for overload in aten.convolution.overloads()})
lowerings.update({getattr(aten.bmm, overload): tuned_bmm for overload in aten.bmm.overloads()})
lowerings.update({getattr(aten._sparse_addmm, overload): sparse_addmm for overload in aten._sparse_addmm.overloads()})
lowerings.update({getattr(aten._unsafe_index, overload): custom_unsafe_index for overload in aten._unsafe_index.overloads()})

lowerings.update({getattr(aten._foobar, overload): custom_foobar for overload in aten._foobar.overloads()})
# 설정에 따라 max_pool2d_with_indices를 커스텀 구현으로 교체할 수 있습니다. (타이밍/측정용 풀링 구현)
if extension_config.CONFIG_USE_TIMING_POOLING:
    lowerings.update({getattr(aten.max_pool2d_with_indices, overload): custom_maxpool for overload in aten.max_pool2d_with_indices.overloads()}) # FIXME: maxpool should be implemented as a template