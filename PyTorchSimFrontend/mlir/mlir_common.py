import dataclasses
import math
from dataclasses import dataclass
from typing import Dict
from typing import List
from collections import defaultdict
from functools import reduce
from operator import mul
import torch
from torch._inductor.codegen import common
from torch._inductor.codegen import cpp
from torch._inductor.virtualized import V
from torch._inductor.ir import MultiOutputLayout
from torch._inductor.dependencies import MemoryDep, StarDep, WeakDep
from torch.utils._sympy.functions import ModularIndexing, FloorDiv, Mod
import sympy
import contextlib

from typing import Callable

import sympy

import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from torch._inductor.utils import (
    free_symbol_startswith,
    get_sympy_Expr_dtype,
    IndentedBuffer,
    sympy_subs,
    sympy_symbol,
    unique,
)
from PyTorchSimFrontend import extension_config
from PyTorchSimFrontend import extension_codecache
schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")

DTYPE_TO_MLIR = {
    torch.float32: "f32",
    torch.float64: "f64",
    torch.float16: "f16",
    torch.int64: "i64",
    torch.int32: "i32",
    torch.int16: "i16",
    torch.int8: "i8",
    torch.uint8: "i8",
    torch.bool: "i8",
    torch.bfloat16: "bf16",
}

MLIR_TO_DTYPE = {
    "f32": torch.float32,
    "f64": torch.float64,
    "f16": torch.float16,
    "i64": torch.int64,
    "i32": torch.int32,
    "i16": torch.int16,
    "i8":  torch.int8,
    "bf16": torch.bfloat16,
}

DTYPE_TO_C = {
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "half",
    torch.int64: "int64_t",
    torch.int32: "int32_t",
    torch.int16: "int16_t",
    torch.int8: "int8_t",
    torch.uint8: "uint8_t",
    torch.bool: "uint8_t",
    torch.bfloat16: "bfloat16",
}

MLIR_TO_BIT = {
    "i1": 1,
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "f16": 16,
    "f32": 32,
    "f64": 64,
    "bf16": 16,
    "index": 64
}

DTYPE_LOWP_FP = [
    torch.bfloat16,
    torch.float16,
]

MLIR_INF = {
    "inf" : {
        "f32" : 0x7F800000,
        "f64" : 0x7FF0000000000000
    },
    "-inf" : {
        "f32" : 0xFF800000,
        "f64" : 0xFFF0000000000000
    },
    "nan" : {
        "f32" : 0x7FC00000,
        "f64" : 0x7FF8000000000000
    }
}

class ParallelLoopBuffer(IndentedBuffer):
    def indent(self, offset=1, attribute="", suffix=""):
        @contextlib.contextmanager
        def ctx():
            for _ in range(offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(-offset):
                if suffix:
                    self.writeline(suffix)
                self._indent -= 1
                self.writeline("} " + attribute)
            yield
            for _ in range(-offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(offset):
                if suffix:
                    self.writeline(suffix)
                self._indent -= 1
                self.writeline("} " + attribute)

        return ctx()

class RecompileSignal(BaseException):
    """
    Exception raised when a recompilation of a kernel or code block is required.
    """
    def __init__(self, message="Recompilation requested."):
        self.message = message
        super().__init__(self.message)

class MLIRKernelArgs(common.KernelArgs):
    """MLIR 전용 커널 인자 헬퍼.

    역할: 그래프의 버퍼/상수/사이즈 정보를 수집하여 MLIR 함수 시그니처에 맞춘
    arg 정의와 호출 인자 목록을 생성합니다. 또한 인자 타입(in/out/inout/var)을
    비트플래그로 관리합니다.
    """
    MLIR_ARGS_IN = 0x01
    MLIR_ARGS_OUT = 0x02
    MLIR_ARGS_INOUT = 0x04
    MLIR_ARGS_VAR = 0x08

    def __init__(self, tile_row=None, tile_col=None):
        # 부모 클래스 초기화 및 MLIR 전용 타일 정보 보관
        super().__init__()
        self.tile_row = tile_row
        self.tile_col = tile_col

    @staticmethod
    def is_mlir_arg_in(value):
        """값이 '입력' 혹은 'inout' 타입인지 판별합니다.

        왜 필요한가: 인자 분류는 코드 생성(읽기 전용/쓰기 포함)에 영향을 줍니다.
        """
        return (MLIRKernelArgs.MLIR_ARGS_IN & value) | (MLIRKernelArgs.MLIR_ARGS_INOUT & value)

    @staticmethod
    def is_mlir_arg_out(value):
        """값이 '출력' 혹은 'inout' 타입인지 판별합니다."""
        return (MLIRKernelArgs.MLIR_ARGS_OUT & value) | (MLIRKernelArgs.MLIR_ARGS_INOUT & value)

    @staticmethod
    def is_mlir_arg_inout(value):
        """값이 'inout' 타입(입출력)인지 판별합니다."""
        return MLIRKernelArgs.MLIR_ARGS_INOUT & value

    @staticmethod
    def get_mlir_shape(info):
        """dtype/numel 정보를 받아 MLIR memref shape 문자열을 생성합니다."""
        tensor_type = DTYPE_TO_MLIR[info[0]]
        return f"memref<{info[1]}x{tensor_type}>"

    def mlir_argdefs(self, extra_node=dict()):
        """그래프의 버퍼/상수/추가 노드 정보를 수집하여
        MLIR 인자 정의(arg_defs), 호출인자(call_args), 인자 속성(arg_attributes)
        및 버퍼 메타 정보(buffer_types)를 반환합니다.

        왜 필요한가: wrapper 코드와 MLIR 함수 선언을 일관되게 생성하기 위해
        모든 인자 정보를 통합하여 제공해야 합니다.
        """
        buffer_types = {}
        # 그래프에 존재하는 버퍼들을 순회하여 메타 정보 수집
        for x in V.graph.buffers:
            if not isinstance(x.layout, MultiOutputLayout): # FIXME: MultiOutputLayout should be handled
                buffer_types[x.get_name()] = [x.get_dtype(), x.get_numel(), x.get_size(), x.get_stride()]
        # 그래프 입력(심볼릭 포함) 처리
        for name, val in V.graph.graph_inputs.items():
            if isinstance(val, sympy.Expr):
                buffer_types[name] = [get_sympy_Expr_dtype(val), 1, [1], [1]]
            else:
                buffer_types[name] = [val.get_dtype(), val.get_numel(), val.get_size(), val.get_stride()]
        # 상수/추가 노드 정보 병합
        buffer_types.update(
            {name: [val.dtype, 1, [1], [1]] for name, val in V.graph.constants.items()}
        )
        buffer_types.update(
            {name: [val.get_dtype(), val.get_numel(), val.get_size(), val.get_stride()] for name, val in extra_node.items()}
        )

        call_args = []
        arg_defs = []
        arg_attributes = []
        def set_info(outer, inner, arg_type):
            # outer: 실제 그래프 이름, inner: MLIR 내부 이름(%X)
            mlir_shape = self.get_mlir_shape(buffer_types[outer])
            arg_defs.append(f"%{inner}: {mlir_shape}")
            call_args.append(outer)
            arg_attributes.append([outer] + [[arg_type] + buffer_types[outer]])

        # inplaced, input, output, sizevar 등 카테고리별로 인자 등록
        for inplaced in unique(self.inplace_buffers.values()):
            if self._buffer_is_marked_removed(inplaced):
                continue
            outer = inplaced.other_names[-1]
            inner = inplaced.inner_name
            set_info(outer, inner, self.MLIR_ARGS_INOUT)
        for outer, inner in self.input_buffers.items():
            if outer in self.inplace_buffers:
                continue
            set_info(outer, inner, self.MLIR_ARGS_IN)
        for outer, inner in self.output_buffers.items():
            if outer in self.inplace_buffers or self._buffer_is_marked_removed(inner):
                continue
            set_info(outer, inner, self.MLIR_ARGS_OUT)
        for outer, inner in self.sizevars.items():
            set_info(outer, inner, self.MLIR_ARGS_VAR)
        return arg_defs, call_args, arg_attributes, buffer_types

class VectorLaneMapping():
    """Vector lane (vlane) 관련 매핑 정보를 관리하는 유틸리티 클래스.

    역할: 주어진 타일을 vlane 단위로 어떻게 분할할지(vlane split axis/stride, 사용 vlane 수 등)를 계산
    하여 각 lane 당 처리량(타일 크기/stride 등)과 벡터화 크기를 결정합니다.
    """
    def __init__(self, vector_lane: int, forced_vec_size: int, vlane_split_axis: int, vlane_stride: int):
        # 하드웨어/매핑 관련 파라미터 보관
        self.vector_lane = vector_lane
        self.vlane_split_axis = vlane_split_axis
        self.vlane_stride = vlane_stride
        self.forced_vec_size = forced_vec_size

    def get_used_vlane(self, tile_size: list[int]):
        """타일 크기에서 실제로 사용될 vlane 수를 계산.

        계산: split_axis 차원의 크기 / vlane_stride를 올림한 값과 전체 vector_lane 중 작은 값을 선택.
        이유: 타일의 해당 축이 vlane 단위로 어떻게 분배되는지 판단하여 자원 할당을 결정하기 위함.
        """
        return min(
            math.ceil(tile_size[self.vlane_split_axis] / self.vlane_stride),
            self.vector_lane
        )

    def get_tile_size_per_lane(self, tile_size: list[int]):
        """타일을 lane당 단위로 나눴을 때의 per-lane 타일 크기를 반환.

        필요한 이유: per-lane 계산량과 메모리 요구량 추정을 위해 각 lane의 타일 크기 정보를 얻어야 함.
        """
        per_lane = tile_size.copy()
        used = self.get_used_vlane(tile_size)
        if self.vlane_split_axis < 0 or self.vlane_split_axis >= len(per_lane):
            raise AssertionError("Not allowed split_axis")
        per_lane[self.vlane_split_axis] = math.ceil(per_lane[self.vlane_split_axis] / used)
        return per_lane

    def get_numel_per_lane(self, tile_size: list[int]):
        # per-lane 타일의 원소 수(=연산량)를 계산
        return math.prod(self.get_tile_size_per_lane(tile_size))

    def get_tile_stride_per_lane(self, tile_size: list[int], tile_stride: list[int]):
        """원래 타일 stride를 per-lane stride로 변환.

        이유: 메모리 접근 패턴을 lane 단위로 재계산하여 DMA/로컬 인덱싱을 조정하기 위함.
        """
        tile_stride = tile_stride.copy()  # original strides
        get_tile_size_per_lane = self.get_tile_size_per_lane(tile_size)
        coeff = tile_size[self.vlane_split_axis]//get_tile_size_per_lane[self.vlane_split_axis]

        # Per-lane로 전파할 때 필요한 stride 보정 수행
        for i in range(len(tile_stride)):
            if tile_stride[i] > tile_stride[self.vlane_split_axis]:
                tile_stride[i] = tile_stride[i] // coeff
        return tile_stride

    def get_compute_vec_size(self, tile_size: list[int], reduction_numel: int, nr_rdim: int) -> int:
        """계산에 사용할 벡터화(또는 SIMD) 단위 크기를 결정.

        - forced_vec_size가 설정되어 있으면 강제값을 반환.
        - reduction이 있는 경우 제약을 고려해 적절한 분할 크기를 계산.
        - 그렇지 않으면 stride 단위를 기준으로 8/4/2 배수 중 적절한 크기를 선택.

        목적: 벡터 연산의 폭을 결정하여 코드 생성 시 vector load/store 및 연산 단위를 맞추기 위함.
        """
        if self.forced_vec_size is not None:
            return self.forced_vec_size

        per_lane = self.get_numel_per_lane(tile_size)
        stride = self.vlane_stride
        if nr_rdim:
            val = per_lane // max(reduction_numel, 1)
            for mult in [8, 4, 2]:
                if per_lane >= val * mult:
                    return val * mult
            return val
        for mult in [8, 4, 2]:
            if (per_lane // stride) >= mult:
                return stride * mult
        return stride

class TileAdjustMixin():
    def __init__(self):
        self.tail_ratio_threshold = 0.01

    def apply_divisor(self, axis: int, divisor: int, mode: str = "split"):
        """Split or pad a given axis of the tile."""
        old_size = self._tile_size[axis]
        if divisor <= 1:
            return

        padded = math.ceil(old_size / divisor) * divisor
        outer = math.ceil(old_size / divisor)
        inner = divisor

        if mode == "pad":
            self._tile_size[axis] = padded
            self.update_tile_stride()
            return
        elif mode == "split":
            new_sizes = list(self._tile_size)
            new_sizes[axis] = outer
            new_sizes.insert(axis + 1, inner)
            self._tile_size = new_sizes

            old_order_val = self.tile_axis_order[axis]
            new_order = list(self.tile_axis_order)
            new_order.insert(axis + 1, old_order_val + 0.1)
            self.tile_axis_order = [idx for idx, _ in sorted(
                zip(range(len(new_order)), new_order), key=lambda x: x[1]
            )]
            self.update_tile_stride()

            # Adjust split axis for vmap
            if self.vmap.vlane_split_axis > axis:
                self.vmap.vlane_split_axis += 1
            return

        raise ValueError(f"Unknown mode: {mode}. Supported: 'pad', 'split'.")

    def is_dim_dividable(self, dim_sizes: list[int]) -> bool:
        if len(dim_sizes) != len(self._tile_size):
            raise ValueError("dim_sizes must match the tile size dimensions")

        dim_sizes_cpy = list(dim_sizes)
        axis, stride = self.vmap.vlane_split_axis, self.vmap.vlane_stride
        remain = dim_sizes_cpy[axis] % stride
        if remain:
            dim_sizes_cpy[axis] += stride - remain

        return all(d % t == 0 for d, t in zip(dim_sizes_cpy, self._tile_size))

    def adjust_tile_to_divisible(self, dim_sizes: list[int]) -> list[int]:
        """Adjust current tile to be divisible by given dimensions."""
        if len(dim_sizes) != len(self._tile_size):
            raise ValueError("dim_sizes must match the tile size dimensions")

        def _adjust_one(dim_size, tile_size):
            for candidate in range(tile_size, 0, -1):
                if dim_size % candidate == 0:
                    return candidate
            return 1

        candidate_tile_size = [_adjust_one(d, t) for d, t in zip(dim_sizes, self._tile_size)]
        for i in range(len(candidate_tile_size)):
            self.tile_constraint[i].must_divide_dim = True

        axis, stride = self.vmap.vlane_split_axis, self.vmap.vlane_stride
        remain = candidate_tile_size[axis] % stride

        if remain:
            candidate_tile_size[axis] += stride - remain
            self.tile_constraint[axis].must_divide_dim = False
        return candidate_tile_size

    def scale_tile_dim(self, axis, dim_sz, scale_factor=2):
        axis_constrinat = self.tile_constraint[axis]
        current_sz = self._tile_size[axis]
        new_sz = axis_constrinat.adjust(current_sz, int(current_sz * scale_factor), dim_sz)
        self._tile_size[axis] = new_sz
        self.update_tile_stride()
        return current_sz != new_sz

    def decrease_tile_size(self, dim_size):
        tile_size = self._tile_size
        vlane_split_axis, vlane_stride, vector_lane = self.vmap.vlane_split_axis, self.vmap.vlane_stride, self.vmap.vector_lane
        tile_size = list(tile_size)

        # Decrease vlane_split_axis when it is too large
        if tile_size[vlane_split_axis] > 2 * vlane_stride * vector_lane:
            if self.scale_tile_dim(vlane_split_axis, dim_size[vlane_split_axis], scale_factor=0.5):
                return

        for i in range(len(tile_size)):
            if i == vlane_split_axis:
                continue
            if tile_size[i] > 1:
                if self.scale_tile_dim(i, dim_size[i], scale_factor=0.5):
                    return

        # Decrease vlane_split_axis at the end to maximize the vlane usage
        self.scale_tile_dim(vlane_split_axis, dim_size[vlane_split_axis], scale_factor=0.5)
        return

    def trim_large_tail(self, ranges: list[int]):
        for i, (dim_range, tile_range) in enumerate(zip(ranges, self._tile_size)):
            ALPHA = 1.0
            BETA = 0.5
            constraint = self.tile_constraint[i]
            if constraint.fixed:
                continue
            elif constraint.must_divide_dim:
                BETA = 0

            padding_ratio = TileAdjustMixin.get_padding_ratio(tile_range, dim_range)
            if padding_ratio < self.tail_ratio_threshold:
                continue
            best_tile = tile_range
            best_cost = (
                ALPHA * padding_ratio +
                BETA * (dim_range / tile_range)
            )

            min_tile = 1
            for candidate in range(tile_range - 1, min_tile - 1, -1):
                new_candidate = constraint.adjust(tile_range, candidate, dim_range)
                ratio = TileAdjustMixin.get_padding_ratio(new_candidate, dim_range)
                iter_penalty = (dim_range / new_candidate)

                cost = ALPHA * ratio + BETA * iter_penalty
                if cost < best_cost:
                    best_tile, best_cost = new_candidate, cost
            self._tile_size[i] = best_tile

    def select_vlane_axis(self):
        best_vlane_split_axis = 0
        best_used_vlane = math.ceil(self._tile_size[0] / self.vmap.vlane_stride)
        for i, dim in enumerate(self._tile_size[1:len(self._tile_size)-self.nr_rdim]):
            used_vlane = math.ceil(dim / self.vmap.vlane_stride)
            if used_vlane > best_used_vlane:
                best_used_vlane = used_vlane
                best_vlane_split_axis = i+1
        self.vmap.vlane_split_axis = best_vlane_split_axis

    def pad_vlane_tile(self):
        # FIXME. this doesn't follow tile constraints...
        vlane_split_axis, vlane_stride, vector_lane = self.vmap.vlane_split_axis, self.vmap.vlane_stride, self.vmap.vector_lane
        used_vlane = min(math.ceil(self._tile_size[vlane_split_axis] / vlane_stride), vector_lane)
        padded_size = used_vlane * vlane_stride
        self._tile_size[vlane_split_axis] = math.ceil(self._tile_size[vlane_split_axis] / padded_size) * padded_size

    def apply_constraints(self, constraints, ranges):
        for idx, (axis_constraints, axis_size) in enumerate(zip(constraints.values(), ranges)):
            for const in axis_constraints:
                if const.args[1] == 1:
                    continue
                divider = int(const.args[1])

                if not self.tile_constraint[idx].fixed:
                    self.tile_constraint[idx].fixed = True
                    self._tile_size[idx] = divider
                elif self.tile_constraint[idx].fixed and self._tile_size[idx] > divider:
                    self._tile_size[idx] = divider
        self.update_tile_stride()

    @staticmethod
    def init_tile_size(ranges, vlane_stride, vector_lane):
        nr_dim = len(ranges)
        tile_size = [1] * nr_dim
        if len(tile_size) == 2:
            tile_size[-1] = vlane_stride * vector_lane
            tile_size[-2] = 2 * vector_lane
        elif len(tile_size) == 0: # Scalar
            tile_size = [1]
            ranges = [1]
        elif len(tile_size) == 1 and ranges[0]==1:
            tile_size[0] = 1
        elif len(tile_size) == 1:
            tile_size[0] = 2 * vlane_stride * vector_lane
        elif len(tile_size) == 3:
            tile_size[-1] = vector_lane
            tile_size[-2] = 4 * vector_lane
            tile_size[-3] = 2
        elif len(tile_size) == 4:
            tile_size[-1] = vector_lane
            tile_size[-2] = 4 * vector_lane
            tile_size[-3] = 2
            tile_size[-4] = 1
        else:
            raise NotImplementedError("dummy tile size fail!")
        return tile_size

    @staticmethod
    def get_padding_ratio(tile_range: int, dim_range: int) -> float:
        if tile_range <= 0 or dim_range <= 0:
            raise ValueError("tile_range and dim_range must be positive integers")
        tail = dim_range % tile_range
        padding = (tile_range - tail) % tile_range
        return float(padding / dim_range)

@dataclass
class TileConstraint:
    multiple_of: int = 1
    must_divide_dim: bool = False
    fixed: bool = False

    def adjust(self, old: int, new: int, dim: int) -> int:
        if self.fixed:
            return old # Fixed tile size

        tail = new % self.multiple_of
        new -= tail
        if not self.must_divide_dim:
            return max(new, self.multiple_of)

        while new > 0:
            if dim % new == 0:
                return new
            new -= self.multiple_of
        raise extension_codecache.TileSizeError("Cannot find suitable tile size under the given constraints.")

class MLIRMultiDimTile(TileAdjustMixin):
    """다차원 타일 설명자(타일 크기/stride/벡터 매핑 등).

    역할: 타일의 크기, stride, axis 순서 및 vlane 매핑 정보를 보관하고,
    코드 생성 시 MLIR에서 사용할 형태로 변환하는 헬퍼를 제공합니다.
    """
    def __init__(self, tile_size, vector_lane, vlane_split_axis=None, vlane_stride=None, forced_vec_size=None):
        super().__init__()
        # 식별자 이름
        self.name = ""
        # 내부 타일 사이즈 리스트
        self._tile_size = list(tile_size)
        # 계산된 stride(초기에는 None, update_tile_stride로 설정)
        self._tile_stride = None
        # 각 축에 대한 제약 정보(TileConstraint 객체 목록)
        self.tile_constraint = [TileConstraint(vlane_stride) for _ in tile_size]
        # 기본 축 순서
        self.tile_axis_order = list(range(len(tile_size)))
        self.update_tile_stride()

        # Vector lane 매핑 설정 보관
        self.vmap = VectorLaneMapping(
            vector_lane=vector_lane,
            forced_vec_size=forced_vec_size,
            vlane_split_axis=vlane_split_axis,
            vlane_stride=vlane_stride
        )

        # implicit dim, reduction 깊이 등 추가 메타
        self.implicit_dim_size = None
        self.nr_rdim = 0
        self.offset = sympy.Integer(0) # DRAM offset을 sympy로 표현

    def set_name(self, name: str): self.name = name
    def get_name(self) -> str: return self.name
    def get_tile_size(self): return list(self._tile_size)
    def get_tile_stride(self): return list(self._tile_stride)
    def get_numel(self) -> int :return math.prod(self._tile_size)
    def get_nr_dim(self) -> str: return len(self._tile_size)
    def get_reduction_numel(self): return reduce(mul, self.get_tile_size()[-1*self.nr_rdim:], 1)

    def set_tile_size(self, tile_size, tile_axis_order=None, constraints=None):
        # 타일 크기/순서/제약을 적용 후 stride를 갱신
        self._tile_size = list(tile_size)
        self.tile_axis_order = list(range(len(tile_size))) if tile_axis_order is None else tile_axis_order
        self.update_tile_stride()

    def set_tile_size_stride(self, tile_size, tile_stride):
        # 타일과 stride를 직접 설정할 때 사용
        self._tile_size = list(tile_size)
        self._tile_stride = list(tile_stride)

    def update_tile_stride(self):
        """타일 사이즈와 axis order로부터 각 축에 대한 stride를 계산.

        이유: 각 차원의 stride는 메모리 인덱싱 및 DMA 인코딩에 필요합니다.
        """
        strides = [1] * len(self._tile_size)
        init = 1

        original_indices = list(range(len(self.tile_axis_order)))
        sorted_pairs = sorted(
            zip(self.tile_axis_order, self._tile_size, original_indices),
            key=lambda x: x[0], reverse=True
        )
        for _, size, original_indices in sorted_pairs:
            strides[original_indices] = init
            init *= size
        self._tile_stride = strides

    def get_dim_size(self, index):
        # 정수 인덱스 또는 'indexN' 형태(sympy 변수)를 받아 해당 축 크기를 반환
        if isinstance(index, int):
            return self._tile_size[index]
        elif "index" in str(index):
            return self._tile_size[int(str(index)[5:])]
        raise NotImplementedError("Unsupported format of index")

   # Vector mapping delegation - 내부 vmap으로 위임
    def get_tile_size_per_lane(self): return self.vmap.get_tile_size_per_lane(self._tile_size)
    def get_used_vlane(self): return self.vmap.get_used_vlane(self._tile_size)
    def get_numel_per_lane(self): return self.vmap.get_numel_per_lane(self._tile_size)
    def get_tile_stride_per_lane(self): return self.vmap.get_tile_stride_per_lane(self._tile_size, self._tile_stride)
    def get_compute_vec_size(self): return self.vmap.get_compute_vec_size(self._tile_size, self.get_reduction_numel(), self.nr_rdim)

    # Helper functions for codegen
    def get_mlir_shape(self, dtype):
        # MLIR memref shape 문자열을 생성(예: memref<16x8xf32, 1>)
        shape = "x".join([str(dim) for dim in self._tile_size])
        return f"memref<{shape}x{dtype}, 1>"

    def get_mlir_vshape(self, mlir_dtype):
        # 벡터 타입 문자열(벡터화 크기 > 1인 경우 vector<...> 형식)
        return f"vector<{self.get_compute_vec_size()}x{mlir_dtype}>" if self.get_compute_vec_size() > 1 else f"{mlir_dtype}"

class MLIRWrapperKenrelGroup(cpp.KernelGroup):
    """MLIR용 래퍼 커널 그룹. 인자와 타일 정보를 보관하는 컨테이너.

    역할: 기존 C++/CUDA용 KernelGroup을 확장하여 MLIR 전용 args와 tile descriptor를 갖도록 함.
    wrapper/렌더러가 타일 정보를 읽고 코드생성에 사용할 수 있도록 중앙에서 관리합니다.
    """
    def __init__(self):
        super().__init__()
        # MLIR 전용 인자 관리 객체
        self.args = MLIRKernelArgs()
        # 코드 생성 시 사용될 MLIRMultiDimTile 객체(초기에는 None)
        self.tile_desc : MLIRMultiDimTile = None

    def set_tile_info(self, tile_desc : MLIRMultiDimTile):
        # 외부에서 계산된 타일 설명자를 등록
        self.tile_desc = tile_desc

class BaseMLIRHardwareInfo():
    """하드웨어 관련 기본 정보 보관용 베이스 클래스.

    역할: VPU/스패드/정밀도/코어 수 등 코드 생성 시 참조되는 하드웨어 파라미터를 중앙화합니다.
    확장자가 원하는 하드웨어 설정을 여기서 정의하여 다른 컴포넌트가 일관되게 사용하도록 합니다.
    """
    def __init__(self):
        # Default HW setting (extension_config에서 값을 읽어 설정)
        # vector_lane: VPU의 lane 수(벡터 연산 폭의 단위)
        self.vector_lane = extension_config.vpu_num_lanes
        # spad_info: 스패드(로컬 메모리) 크기/설정 정보
        self.spad_info = extension_config.CONFIG_SPAD_INFO
        # 정밀도(byte 단위 등) - e.g., f32 -> 4 bytes
        self.precision = extension_config.CONFIG_PRECISION
        # 병렬 코어 수
        self.num_cores = extension_config.CONFIG_NUM_CORES
        # 벡터 레지스터 길이(bits)
        self.vlen = extension_config.vpu_vector_length_bits

class BaseMLIRKernel(common.Kernel, BaseMLIRHardwareInfo):
    """MLIR 코드 생성의 공통 기반 클래스.

    역할: MLIR 전용 코드 버퍼, CSE, 변수/버퍼 메타, 루프 정보 및 재컴파일 로직을 제공하여
    실제 커널별 구현이 이를 활용하도록 공통 기능을 캡슐화합니다.
    """
    newvar_prefix = "%"
    suffix = ""
    overrides = None
    load_format = None
    store_format = None

    def __init__(self, kernel_group, reason=None):
        """기본 상태(버퍼, 루프, CSE 등)를 초기화.

        reason: 재컴파일 원인을 전달(recodegen)하여 타일 조정/재시도 로직에 사용.
        """
        super().__init__(kernel_group.args)
        self.kernel_group = kernel_group
        # 루프/범위 관련 상태
        self.call_ranges = None
        self.ranges = None
        self.reduction_depth = None
        self.itervars = None
        # 코드 버퍼: 벡터 연산 본문, 리덕션 접미사 등
        self.vector_compute = IndentedBuffer()
        self.reductions_suffix = IndentedBuffer()
        self.cse = common.CSE(self.newvar_prefix, self.suffix)
        # MLIR SSA 변수 정보 추적기
        self.var_info = {} # MLIR variable info
        self.buffer_types : dict = None # format: dtype, numel, size, stride
        # compute index 이름 및 루프 레벨 설정
        self.compute_idx = "compute_idx"
        self.compute_body_loop = LoopLevel(self.compute_idx, 1)
        self.prologue_compute_body_loop = LoopLevel(self.compute_idx, 1)
        # 재컴파일 이유 (예: spad overflow 등)를 저장
        self.recodegen = reason # spad overflow, tile size, vlane stride
        self.stop_autotune = False

    def set_ranges(self, lengths, reduction_lengths):
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(
                reduction_lengths
            ), f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [sympy.Symbol(f"index{n}") for n in range(len(self.ranges))]
            self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

    def get_nr_rdim(self):
        return len(self.itervars[self.reduction_depth:])

    def load(self, name: str, index: sympy.Expr):
        raise NotImplementedError()

    def store_reduction(self, name, index, value):
        raise NotImplementedError()

    def store(self, name, index, value, mode=None):
        raise NotImplementedError()

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise NotImplementedError()

    def indirect_indexing(self, index_var, size, check):
        raise NotImplementedError()

    def codegen_global_init(self):
        raise NotImplementedError()

    def codegen_loops(self):
        raise NotImplementedError()

    def call_kernel(self, kernel_name):
        wrapper = V.graph.wrapper_code
        _, call_args, _, _ = self.kernel_group.args.mlir_argdefs()
       # generate the code to call this
        wrapper.generate_kernel_call(kernel_name, call_args, cuda=False)

    def is_modular_indexing(self, expr):
        return "ModularIndexing" in str(expr)

    def implicit_dim_ops(self, nodes):
        target_patterns = (ModularIndexing, FloorDiv, Mod)
        target_operands = []
        for target_node in nodes:
            for read_operand in target_node.read_writes.reads:
                read_operand: MemoryDep
                if isinstance(read_operand, StarDep) or isinstance(read_operand, WeakDep):
                    continue
                read_index = read_operand.index
                for arg_expr in read_index.args:
                    if arg_expr.atoms(*target_patterns):
                        target_operands.append(read_operand)
        return target_operands

    def extract_dividers(self, implicit_ops):
        # When a specific axis is processed, the key constraint to verify is the divider.
        # The tile size must be forced to match the divider size.
        dim_dividers = defaultdict(set)
        for operand in implicit_ops:
            subs_map = {
                s: sympy.symbols(s.name.replace("c", "index", 1))
                for s in operand.index.free_symbols
            }
            rev_subs_map = {
                sympy.symbols(s.name.replace("c", "index", 1)) : s
                for s in operand.index.free_symbols
            }
            new_index = operand.index.subs(subs_map)
            for arg in new_index.args:
                if len(arg.free_symbols) != 1:
                    raise NotImplementedError("Not supporting this view operation...!")
                if arg.is_Mul and arg.args[0].is_number:
                    arg = arg.args[1]

                if isinstance(arg, ModularIndexing):
                    modular_expr = ModularIndexing(arg.args[0], arg.args[1], arg.args[2])
                    modular_expr.original_expr = arg
                elif arg.is_symbol:
                    modular_expr = ModularIndexing(arg, 1, operand.ranges[rev_subs_map[arg]])
                    modular_expr.original_expr = arg
                elif "//" in str(arg):
                    modular_expr = ModularIndexing(arg.args[0], arg.args[1], operand.ranges[rev_subs_map[arg.args[0]]]//arg.args[1])
                    modular_expr.original_expr = arg
                else:
                    raise NotImplementedError("What is this case?")
                dim_dividers[modular_expr.args[0]].add(modular_expr)
        return dim_dividers

    def compute_tile_size(self, nodes, vars, reduction_vars):
        vlane_split_axis = len(vars) - 1
        vlane_stride = 2 # Set minimum vlane stride

        # Set initial tile size & vector lane mapping
        if self.kernel_group.tile_desc is None:
            tile_size = MLIRMultiDimTile.init_tile_size(self.ranges, vlane_stride, self.vector_lane)
            init_tile_desc = MLIRMultiDimTile(tile_size, self.vector_lane, vlane_split_axis, vlane_stride)
            init_tile_desc.nr_rdim = len(reduction_vars)
            self.kernel_group.set_tile_info(init_tile_desc)

        # Handle edge case
        if len(self.ranges)==1 and self.ranges[0] == 1: # Scalar case 2
            self.kernel_group.tile_desc.vmap.vlane_stride = 1
            self.kernel_group.tile_desc.vmap.vlane_split_axis = 0
        elif vlane_split_axis == -1: # Reduction only case
            self.kernel_group.tile_desc.vmap.vlane_split_axis = 0
            self.kernel_group.tile_desc.vmap.vlane_stride = self.kernel_group.tile_desc.get_tile_size()[0]

        # Handle implict dims. Input operand could be high dimension tensor.
        # Note: https://github.com/PSAL-POSTECH/PyTorchSim/issues/173
        implicit_ops = self.implicit_dim_ops(nodes)
        if implicit_ops:
            tile_constraints = self.extract_dividers(implicit_ops)
            self.kernel_group.tile_desc.apply_constraints(tile_constraints, self.ranges)
            self.kernel_group.tile_desc.implicit_dim_size = tile_constraints

        # Check recodegen reason
        if self.recodegen is not None:
            if self.recodegen == "spad_overflow":
                self.kernel_group.tile_desc.decrease_tile_size(self.ranges)
            elif self.recodegen == "recompile":
                return self.kernel_group.tile_desc
            else:
                raise NotImplementedError(f"Unknown recodegen reason: {self.recodegen}")

        # Adjust tile size & vector lane mapping
        self.kernel_group.tile_desc.trim_large_tail(self.ranges)
        self.kernel_group.tile_desc.select_vlane_axis()
        self.kernel_group.tile_desc.pad_vlane_tile()
        self.kernel_group.tile_desc.update_tile_stride()
        return self.kernel_group.tile_desc

    def codegen_nodes(self, nodes, kernel_name):
        recompile_try = 0
        max_retry_compile = 5
        while True:
            _, (group, reduction_group) = max(
                nodes, key=lambda x: int(x.is_reduction())
            ).group

            # Set node range info
            vars, reduction_vars = self.set_ranges(group, reduction_group)
            tile_desc = self.compute_tile_size(nodes, vars, reduction_vars)
            self.compute_body_loop.size = tile_desc.get_numel_per_lane()
            self.compute_body_loop.step = tile_desc.get_compute_vec_size()
            try:
                _, _, _, self.buffer_types = self.kernel_group.args.mlir_argdefs()
                with self as kernel:
                    for node in nodes:
                        node.run(vars, reduction_vars)
            except RecompileSignal as e:
                recompile_try += 1
                if recompile_try > max_retry_compile:
                    raise RuntimeError("Failed to compile kernel after multiple attempts.")
                # Retry compile nodes
                #print(f"Try recompile({recompile_try}/{max_retry_compile}). Reason: {e}")
                continue
            V.graph.removed_buffers |= self.removed_buffers
            # V.graph.inplaced_to_remove |= self.inplaced_to_remove
            src_code = self.codegen_kernel(kernel_name=kernel_name)
            self.meta_kernel()
            return src_code

    def codegen_kernel(self, kernel_name):
        arg_defs, _, _, _ = self.kernel_group.args.mlir_argdefs()
        arg_defs = ",\n".ljust(25).join(arg_defs)
        code = common.BracesBuffer()

        #TODO:. kernel name custom
        kernel_decl_name = kernel_name if V.graph.cpp_wrapper else "kernel"

        code.splice(self.codegen_global_init())
        code.writeline(f'func.func @{kernel_decl_name}({arg_defs})')
        with code.indent():
            for old, new in self.kernel_group.args.aliases():
                code.writeline(f"auto {old} = {new};")
            # Loop body part
            code.splice(self.codegen_loops())
        return code.getvalue()

    def meta_kernel(self):
        wrapper = V.graph.wrapper_code
        _, _, arg_attributes, _ = self.kernel_group.args.mlir_argdefs()
        wrapper.add_import_once('\nprint(f\'Wrapper Codegen Path = {__file__}\')')
        # Dump loop and load/store information
        wrapper.add_import_once(f"arg_attributes = {arg_attributes}")
        return arg_attributes

    def get_constant_vector(self, expr):
        constant_vector = [[int(expr.coeff(var)),None] for var in self.itervars]
        return constant_vector

    def get_constant_vector2(self, expr):
        # Case 0. symbol ex) index 0
        # Case 1. inner product form ex) 16 * index0 + 1 * index1
        # Case 2. Complicated form ex) 16 * index0 + 8 * (index//4) + (index % 4)
        constant_vector = []
        if expr.is_symbol:
            constant_vector.append(tuple([1, expr]))
            return constant_vector

        for arg in expr.args:
            if arg.is_symbol:
                constant_vector.append(tuple([1,arg]))
                continue
            if len(arg.args) == 0: #TODO: check this
                continue
            if arg.args[0].is_number:
                constant_vector.append(arg.args)
            else:
                constant_vector.append([1, arg])

        return constant_vector

    def find_node_by_name(self, name):
        if name in V.graph.graph_inputs:
            return V.graph.graph_inputs[name]
        else:
            for output_node in V.graph.graph_outputs:
                if output_node.data.name == name:
                    return output_node

    def is_scalar(self, name):
        return self.buffer_types[name][1] == 1

    def roundup_vectorlane(self, size, amp=1):
        return ((size + self.vector_lane - 1) // self.vector_lane) * self.vector_lane * amp

    def register_var_info(self, var, var_info):
        self.var_info[var] = var_info

    def rename_indexing(self, index) -> sympy.Expr:
        # adds the necessary kernel args for index expressions
        # and renames variables in index expressions to kernel arg names
        if isinstance(index, (list, tuple)):
            return [self.rename_indexing(x) for x in index]
        index = V.graph.sizevars.simplify(index)
        sorted_symbols = sorted(index.free_symbols, key=lambda s: s.name)
        replacements = {
            x: self.kernel_group.args.size(x)
            for x in sorted_symbols
            if x.name.startswith("s") or x.name.startswith("ps")
        }
        return sympy_subs(index, replacements)

    def __enter__(self):
        class CSEProxy:
            self.name = "CSEProxy"

            @staticmethod
            def __getattr__(name: str) -> Callable[..., common.CSEVariable]:  # type: ignore[misc]
                def inner(*args, **kwargs):
                    code, ret_info = getattr(parent_handler, name)(*args, var_info=self.var_info)
                    csevar = self.cse.generate(
                        self.compute,
                        code,
                        bounds=ValueRanges.unknown(),
                        assignment=(ret_info[0] is not None)
                    )
                    if ret_info[0] is not None:
                        self.register_var_info(csevar, ret_info)
                        csevar.update_on_args(name, args, kwargs)
                    return csevar

                return inner

            @staticmethod
            def indirect_indexing(index_var, size, check=True):
                # Skip CSE since this doesn't return an expression
                return self.indirect_indexing(index_var, size, check)

            @staticmethod
            def load(name: str, index: sympy.Expr):
                if name in self.cse.invalidated_stores:
                    # A load from an invalidated store requires us to
                    # keep the actual buffer around
                    V.kernel.must_keep_buffers.add(name)
                if free_symbol_startswith(index, "%"):
                    return self.indirect_load(name, index)
                store_cache = self.cse.store_cache
                if name in store_cache:
                    return store_cache[name]
                key = name+str(index)
                if key not in self.cse.cache:
                    result = self.load(name, index)
                    self.cse.cache[key] = result
                return self.cse.cache[key]

            @staticmethod
            def store(name, index, value, mode=None):
                self.store_buffer_names.add(name)
                if mode is None:
                    self.cse.store_cache[name] = value
                    if self.current_node:
                        for other_name in self.current_node.get_mutations():
                            self.cse.store_cache[other_name] = value
                if name not in V.graph.removed_buffers:
                    return self.store(name, index, value, mode=mode)

            @staticmethod
            def store_reduction(name, index, value):
                self.store_buffer_names.add(name)
                self.cse.store_cache[name] = value
                if self.current_node:
                    for other_name in self.current_node.get_mutations():
                        self.cse.store_cache[other_name] = value

                if name not in V.graph.removed_buffers:
                    return self.store_reduction(name, index, value)

            @staticmethod
            def reduction(dtype, src_dtype, reduction_type, value):
                return self.reduction(dtype, src_dtype, reduction_type, value)

            @staticmethod
            def _index_expr(tile_size, buffer, renamed_expression, index):
                return self._index_expr(tile_size, buffer, renamed_expression, index)

            @staticmethod
            def index_expr(index, dtype):
                return self.index_expr(index, dtype)

            @staticmethod
            def bucketize(
                values,
                offsets_name: str,
                offsets_size: sympy.Expr,
                indexing_dtype: torch.dtype,
                right: bool,
            ):
                """
                [Note: Inductor bucketize op]

                Given values (tensor) and offsets_name (reference to the name of a 1D
                tensor), calculate the bucket that each value belongs to.

                e.g. for values [-1, 0, 1, 2, 3, 4, 5, 9], offsets [0, 4, 4, 8], right=True
                return =        [ 0, 1, 1, 1, 1, 3, 3, 4].

                When right == False, bucket i refers to range (offsets[i], offsets[i+1]].
                When right == True,  bucket i refers to range [offsets[i], offsets[i+1]).

                Offsets must be non-decreasing or the result is undefined.
                """
                return self.bucketize(
                    values, offsets_name, offsets_size, indexing_dtype, right
                )

        super().__enter__()
        assert self.overrides
        parent_handler = self.overrides(V.get_ops_handler())
        self.exit_stack.enter_context(V.set_ops_handler(CSEProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self


@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr
    size: sympy.Expr
    start: int = 0
    step: int = 1
    reduction_vars: Dict[str, str] = dataclasses.field(default_factory=dict)
    affine_yield: Dict[str, str] = dataclasses.field(default_factory=dict)

    def lines(self):
        if len(self.reduction_vars):
            acc = ', '.join([f"%{acc.name}" for acc in self.reduction_vars.keys()])
            args = ', '.join([f"%{iter.name} = %{init.name}" for (_, iter, init, _) in self.reduction_vars.values()])
            dtype = ', '.join([f"{dtype}" for (_, _, _, dtype) in self.reduction_vars.values()])
            line = f"{acc} = affine.for %{self.var} = {self.start} to {self.size} step {self.step} iter_args({args}) -> ({dtype})"
        else:
            line = f"affine.for %{self.var} = {self.start} to {self.size} step {self.step}"

        return [line]

    def epilogue_line(self):
        if len(self.affine_yield):
            vars = ', '.join([f"%{name}" for name, _ in self.affine_yield.items()])
            reduced_shapes = ', '.join([f"{shape}" for _, shape in self.affine_yield.items()])
            return f"affine.yield {vars} : {reduced_shapes}"
        return ""

@dataclasses.dataclass
class LoopNest:
    loops: List[LoopLevel]

    def __bool__(self):
        return bool(self.loops)

    def mark_reduction(self, reduction_vars, affine_yield=dict()):
        for loop_depth, loop in enumerate(self.loops):
            loop.reduction_vars = {key: list(val)[:-1] for key, val in reduction_vars.items() if val[-1] == loop_depth}
            loop.affine_yield = {key: val[0] for key, val in affine_yield.items() if val[-1] == loop_depth}

    def mark_parallel(self, par_depth):
        loops = self.loops
        loops[0].parallel = par_depth
        for i in range(1, par_depth):
            loops[i].collapsed = True
        loops[0].simd = loops[par_depth - 1].simd