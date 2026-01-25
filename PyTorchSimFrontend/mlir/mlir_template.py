# mlir_template.py
# MLIR 템플릿 기반의 커널 생성/타일링/매핑 유틸리티들을 포함하는 파일입니다.

import functools  # 함수 헬퍼(예: partial) 사용
import itertools  # 반복자 조합, 순열 등 유틸리티
import textwrap  # 코드 블록 정렬/포맷에 사용
import re  # 정규 표현식 처리
import os  # 파일/디렉터리 조작
import contextlib  # 컨텍스트 매니저 유틸리티
import math  # 수학 함수 (ceil, sqrt 등)
import sympy  # 기호 수학(약수, 분해 등) 유틸리티
from functools import reduce  # 시퀀스 누적 연산에 사용
import operator  # 연산자 함수(곱셈 등) 사용
from collections import OrderedDict  # 순서 유지 딕셔너리

from typing import List, Optional  # 타입 힌트
from unittest.mock import patch  # 테스트나 임시 패치용

# Inductor 내부의 공통 템플릿/유틸 가져오기
from torch._inductor.codegen.common import KernelTemplate, ChoiceCaller, CSE, DeferredLine  # 코드 생성 공통 유틸
from torch._inductor.ir import Buffer, IRNode, TemplateBuffer  # IR 관련 타입
from torch._inductor.select_algorithm import PartialRender  # 부분 렌더링 도우미
from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller  # CUDA 호출러(참조)
from torch._inductor.autotune_process import TensorMeta  # 오토튜닝을 위한 텐서 메타
from torch._inductor.virtualized import V, NullHandler, _ops as ops  # 가상화 유틸
from torch._inductor.utils import IndentedBuffer  # 들여쓰기 지원 버퍼
from torch._inductor.codecache import write_atomic  # 코드 캐시 쓰기 유틸

# 확장(Frontend) 관련 모듈
import PyTorchSimFrontend.extension_codecache as extension_codecache  # 확장된 코드 캐시 구현
from PyTorchSimFrontend.mlir.mlir_autotune import MLIRBenchmarkRequest  # MLIR용 벤치 요청 구조
from PyTorchSimFrontend.mlir.mlir_common import BaseMLIRHardwareInfo  # 하드웨어 관련 정보 베이스
from PyTorchSimFrontend.mlir.mlir_codegen_backend import MLIRKernel, reduction_init, reduction_partial_combine_vec, reduction_combine_vec, is_welford_reduction  # MLIR 코드 생성 백엔드 함수
from PyTorchSimFrontend.mlir.mlir_scheduling import SchedulerNode  # 스케줄링 관련 노드 타입
from torch._inductor.codegen import common  # 공통 코드 생성 유틸

from PyTorchSimFrontend import extension_config  # 확장 설정 로드
from . import mlir_common  # 같은 패키지의 공용 유틸

class IndentedBufferGroup:
    """여러 IndentedBuffer( loads/compute/stores 등)를 그룹화하여 임시로 커널에 적용/복원하는 유틸.

    사용 예: prologue/epilogue 등 특정 블록에서 별도의 버퍼로 코드 생성을 수행한 뒤 원상 복귀.
    """
    def __init__(self, kernel: 'MLIRTemplateKernel', prefix=""):
        # kernel 참조와 여러 목적의 IndentedBuffer를 초기화합니다.
        self.kernel = kernel
        self.body = IndentedBuffer()  # 전체 바디용
        self.loads = IndentedBuffer()  # 로드 라인용
        self.compute = IndentedBuffer()  # 계산 라인용
        self.stores = IndentedBuffer()  # 저장 라인용
        self.applys = IndentedBuffer()  # 후처리용 라인
        self.dma_loads = IndentedBuffer()  # DMA 로드 전용
        self.dma_stores = IndentedBuffer()  # DMA 저장 전용
        self.spad_buffer = IndentedBuffer()  # 스패드 관련 라인
        # CSE(공통 하위식 제거) 인스턴스들: 이름 접두사로 구분
        self.cse = common.CSE("%", "", name_prefix=f"{prefix}")
        self.apply_cse = common.CSE("%", "", name_prefix=f"{prefix}apply")
        # with 블록 진입 전 원래 버퍼들을 저장하기 위한 사전
        self.original_buffers = {}

    def set_buffers(self):
        # 현재 그룹의 버퍼들을 실제 커널의 속성으로 설정하여, 이후 생성되는 코드가 여기에 기록되게 합니다.
        self.kernel.loads = self.loads
        self.kernel.compute = self.compute
        self.kernel.stores = self.stores
        self.kernel.applys = self.applys
        self.kernel.dma_loads = self.dma_loads
        self.kernel.dma_stores = self.dma_stores
        self.kernel.spad_buffer = self.spad_buffer
        self.kernel.cse = self.cse
        self.kernel.apply_cse = self.apply_cse

    def restore_buffers(self):
        # 저장해둔 원래 버퍼들을 복원합니다.
        self.kernel.loads = self.original_buffers['loads']
        self.kernel.compute = self.original_buffers['compute']
        self.kernel.stores = self.original_buffers['stores']
        self.kernel.applys = self.original_buffers['applys']
        self.kernel.dma_loads = self.original_buffers['dma_loads']
        self.kernel.dma_stores = self.original_buffers['dma_stores']
        self.kernel.spad_buffer = self.original_buffers['spad_buffer']
        self.kernel.cse = self.original_buffers['cse']
        self.kernel.apply_cse = self.original_buffers['apply_cse']

    @contextlib.contextmanager
    def as_local(self):
        # 컨텍스트 진입 시 현재 커널의 버퍼들을 저장하고 그룹 버퍼로 교체합니다.
        self.original_buffers = {
            'loads': self.kernel.loads,
            'compute': self.kernel.compute,
            'stores': self.kernel.stores,
            'applys': self.kernel.applys,
            'dma_loads': self.kernel.dma_loads,
            'dma_stores': self.kernel.dma_stores,
            'spad_buffer': self.kernel.spad_buffer,
            'cse': self.kernel.cse,
            'apply_cse': self.kernel.apply_cse,
        }
        try:
            self.set_buffers()  # 그룹 버퍼로 교체
            yield self
        finally:
            self.restore_buffers()  # 종료 시 복원

class MLIRTemplateKernel(MLIRKernel, BaseMLIRHardwareInfo):
    """MLIR 기반 템플릿 커널을 표현하는 핵심 클래스입니다.

    이 클래스는 템플릿 렌더링에 필요한 메타데이터, 루프/타일 정보, CSE, prologue/epilogue 버퍼 그룹 등을 관리합니다.
    """
    def __init__(self,
                 kernel_name,
                 input_nodes,
                 call_size,
                 kernel_group = None,
                 outer_func_name=None,
                 outer_func_render=None,
                 kernel_arg_attributes=None,
                 reason=None) -> None:
        # MLIRKernel 초기화: kernel_group이 주어지지 않으면 기본 Wrapper 그룹 사용
        super().__init__(kernel_group if kernel_group is not None else mlir_common.MLIRWrapperKenrelGroup())
        # 식별자 및 입력/콜 사이즈 저장
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.call_size = call_size
        # 노드 이름과 루프 정보를 위한 컨테이너
        self.named_nodes = {}
        self.loop_info = {}
        # outer function 관련 선택적 정보
        self.outer_func_name = outer_func_name
        self.outer_func_render = outer_func_render
        # 커널 인자 속성을 외부에서 주입 가능
        self.kernel_arg_attributes = kernel_arg_attributes
        # 렌더 후크, 버퍼 이름, 렌더 옵션
        self.render_hooks = OrderedDict()
        self.buffer_names = dict()
        self.render_options = dict()
        # 타일/루프 관련 변수
        self.tile_size = []
        self.loop_size = None
        # CSE(공통 하위식 제거) 인스턴스들: 맵/상수/할당 식별자에 사용
        self.map_cse = CSE("#", self.suffix, name_prefix="t_map")
        self.const_cse = CSE(self.newvar_prefix, self.suffix, name_prefix="t_const")
        self.alloc_cse = CSE(self.newvar_prefix, self.suffix, name_prefix="t_alloc")
        # Prologue/Epilogue에서 별도 버퍼 관리를 위한 그룹
        self.prologue_buffer_group = IndentedBufferGroup(self, prefix="prologue_")
        self.epilogue_buffer_group = IndentedBufferGroup(self, prefix="epilogue_")
        # 전역 변수와 예외 노드 저장
        self.global_vars = IndentedBuffer()
        self.exception_nodes = {}
        # Reduction 관련 상태와 버퍼
        self.reduction_epilogue_suffix = IndentedBuffer()
        self.reduction_fusion = False
        self.reduction_body_loop = None
        self.reduction_buffer_idx = 0
        self.reduction_info = {}
        self.reduction_epilogue_result = {}
        self.reduction_mean = []
        # 차원(alias) 정보 및 이유(reason)
        self.dim_aliasing = {}
        self.reason = reason

    def reset(self, reason):
        """커널 상태를 주어진 reason으로 재초기화합니다.

        테스트나 재사용 시 인스턴스를 초기 상태로 되돌리기 위해 사용합니다.
        """
        self.__init__(
            self.kernel_name, self.input_nodes,
            self.call_size, self.kernel_group,
            self.outer_func_name, self.outer_func_render,
            self.kernel_arg_attributes, reason
        )

    def add_loop_info(self, mat_size, tile_size):
        """행렬 및 타일 크기로부터 각 루프 인덱스의 [start, end, stride] 정보를 생성하여 저장합니다.

        mat_size: 전체 루프 범위, tile_size: 각 차원에서의 타일 크기(스트라이드)
        """
        for idx, (loop_size, stride) in enumerate(zip(mat_size, tile_size)):
            # index0, index1, ... 형태의 키로 루프 정보를 저장
            self.loop_info[f"index{idx}"] = [0, loop_size, stride]

    def gemmini_gemm_mapping(self, M, N, K):
        spad_size = self.spad_info["spad_size"] * self.vector_lane
        num_cores = self.num_cores
        precision = self.precision
        dim_I, dim_J, dim_K = M, N, K
        dim = self.vector_lane

        # split spad into 3/4 for input and 1/4 for output (only for mapping)
        # TODO: 3/4 and 1/4 are arbitrary numbers. We should find a better way to split the spad (auto-tune?)
        max_spad_rows = (spad_size * 3 // 4) // (dim * precision * 2) # 4 bytes per element, double buffer
        max_acc_rows = (spad_size // 4) // (dim * 4 * 2) # 4 bytes per element, double buffer

        dim_I_padded = (dim_I // dim + (dim_I % dim != 0)) * dim
        dim_J_padded = (dim_J // dim + (dim_J % dim != 0)) * dim
        dim_K_padded = (dim_K // dim + (dim_K % dim != 0)) * dim

        db_partitions_rows = max_spad_rows // 2
        db_mats_in_partition = db_partitions_rows // dim
        db_mats_in_acc = max_acc_rows // dim
        db_max_tile_i_j = int(math.sqrt(db_mats_in_acc))
        db_max_tile_k = db_mats_in_partition // db_max_tile_i_j

        tile_I = min(dim_I_padded // dim, math.ceil(dim_I / (db_max_tile_i_j * dim)))
        tile_J = min(dim_J_padded // dim, math.ceil(dim_J / (db_max_tile_i_j * dim)))
        tile_K = min(dim_K_padded // dim, math.ceil(dim_K / (db_max_tile_k * dim)))

        num_tiles = tile_I * tile_J
        if num_tiles < num_cores:
            increase_tile = math.ceil(num_cores / num_tiles)
            if dim_J > dim_I and dim_J > num_cores:
                tile_J *= increase_tile
            elif dim_I > dim_J and dim_I > num_cores:
                tile_I *= increase_tile
            num_tiles = tile_I * tile_J
        if num_tiles % num_cores != 0:
            increase_tile = num_tiles % num_cores
            if dim_J > dim_I and dim_J > num_cores:
                tile_J += increase_tile
            elif dim_I > dim_J and dim_I > num_cores:
                tile_I += increase_tile

        inner_I = math.ceil(dim_I_padded / tile_I)
        inner_J = math.ceil(dim_J_padded / tile_J)
        inner_K = math.ceil(dim_K_padded / tile_K)

        inner_I -= inner_I & (dim) - 1
        inner_J -= inner_J & (dim) - 1
        inner_K -= inner_K & (dim) - 1

        tile_I = math.ceil(dim_I / inner_I)
        tile_J = math.ceil(dim_J / inner_J)
        tile_K = math.ceil(dim_K / inner_K)

        return inner_I, inner_J, inner_K

    def gemm_combination_mapping(self, M, N, K, n_extra_node=0, n_prologue_node=0, pad_k=True, min_tile=False, is_conv=False):
        """GEMM용 타일 후보들을 생성하고 휴리스틱으로 우수 후보를 선택합니다.

        고려 항목: 스패드 사용량, lane 당 사용량, weight reuse, 최소 타일 수 등
        """
        tile_candidates = []
        # 스패드/레인/정밀도 정보
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2 # double buffer을 고려한 최대 사용 가능 스패드
        max_spad_per_lane = spad_size_per_lane // 2 # lane 당 최대 스패드
        minimum_n_tile = self.num_cores if min_tile else 1
        # 패딩 팩터 결정: 벡터 lane 단위로 패딩하거나 기본값 8을 사용
        m_pad_factor = self.vector_lane if M > self.vector_lane else 8
        n_pad_factor = self.vector_lane if N > self.vector_lane else 8
        k_pad_factor = self.vector_lane if K > self.vector_lane else (8 if pad_k else 1)
        K = max(K, 8)
        # 차원을 패딩하여 정렬 단위를 맞춤
        M_padded = ((M + m_pad_factor - 1) // m_pad_factor) * m_pad_factor
        N_padded = ((N + n_pad_factor - 1) // n_pad_factor) * n_pad_factor
        K_padded = ((K + k_pad_factor - 1) // k_pad_factor) * k_pad_factor
        indexI, indexJ, indexK = (M_padded // self.vector_lane, N_padded // self.vector_lane, K_padded // self.vector_lane)

        max_used_spad_size = 0
        mapping = (self.vector_lane, self.vector_lane, self.vector_lane)  # 기본 매핑
        # 타일 분할 후보의 약수를 이용하여 후보 범위를 만듭니다.
        tile_M_range = sympy.divisors(indexI) if M > self.vector_lane else [1]
        tile_N_range = sympy.divisors(indexJ) if N > self.vector_lane else [1]
        tile_K_range = sympy.divisors(indexK) if K > self.vector_lane else [1]
        maximize_i_j = 1 # weight reuse를 극대화하기 위한 보조 변수
        for k in tile_K_range:  # K 차원의 타일 후보 반복 (각 k는 factor)
            # tile_K: 실제 타일의 K 크기. K가 vector_lane보다 큰 경우 벡터 레인 단위로 확장
            tile_K = k * self.vector_lane if K > self.vector_lane else K_padded
            for i in tile_M_range:  # M 차원 타일 후보 반복
                # tile_M: M 차원의 실제 타일 크기 (vector lane 단위 또는 패딩된 값)
                tile_M = i * self.vector_lane if M > self.vector_lane else M_padded
                for j in tile_N_range:  # N 차원 타일 후보 반복
                    # tile_N: N 차원의 실제 타일 크기
                    tile_N = j * self.vector_lane if N > self.vector_lane else N_padded
                    # 다음으로 각 후보에 대해 필요한 스패드 사용량(입력, 가중치, 출력 포함)을 추정합니다.
                    # used_spad_size는 전체 스패드 사용량(바이트 단위, precision을 곱함)을 의미합니다.
                    used_spad_size = (tile_M * tile_K * (1 + n_prologue_node) + tile_K * tile_N + tile_M * tile_N * (1 + n_extra_node)) * self.precision
                    # lane 당 가중치/입력/출력의 크기를 계산하여 lane 분산 관점에서의 사용량을 추정합니다.
                    weight_size_per_lane = self.get_spad_size_per_lane(tile_K, tile_N)  # 가중치 크기 per lane
                    input_size_per_lane = self.get_spad_size_per_lane(tile_M * (1 + n_prologue_node), tile_K)  # 입력 크기 per lane
                    output_size_per_lane = self.get_spad_size_per_lane(tile_M * (1 + n_extra_node), tile_N)  # 출력 크기 per lane
                    # lane 당 사용량들을 합쳐 실제 lane 단위로 필요한 스패드 사용량을 계산합니다.
                    used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane + output_size_per_lane) * self.precision
                    check_spad_size = (used_spad_size < max_spad_size and used_spad_size_per_lane < max_spad_per_lane)
                    if check_spad_size:
                        # 디렉터리/파일에 후보를 기록하여 외부 검증/수집에 사용합니다.
                        dir_path = f"{extension_config.CONFIG_TORCHSIM_DIR}/validation/gemm_candidates"
                        os.makedirs(dir_path, exist_ok=True)
                        file_path = f"{dir_path}/gemm_{M}_{K}_{N}.txt"
                        line_to_write = f"{tile_M} {tile_K} {tile_N}\n"
                        try:
                            with open(file_path, "r") as f:
                                lines = f.readlines()
                        except FileNotFoundError:
                            lines = []
                        if line_to_write not in lines:
                            with open(file_path, "a") as f:
                                f.write(line_to_write)

        # 휴리스틱 탐색: 후보들을 평가하여 최적 후보를 선정
        for k in tile_K_range: # heuristic search
            tile_K = k * self.vector_lane if K > self.vector_lane else K_padded
            for i in tile_M_range:
                tile_M = i * self.vector_lane if M > self.vector_lane else M_padded
                for j in tile_N_range:
                    tile_N = j * self.vector_lane if N > self.vector_lane else N_padded
                    used_spad_size = (tile_M * tile_K * (1 + n_prologue_node) + tile_K * tile_N + tile_M * tile_N * (1 + n_extra_node)) * self.precision
                    weight_size_per_lane = self.get_spad_size_per_lane(tile_K, tile_N)
                    input_size_per_lane = self.get_spad_size_per_lane(tile_M * (1 + n_prologue_node), tile_K)
                    output_size_per_lane = self.get_spad_size_per_lane(tile_M * (1 + n_extra_node), tile_N)
                    used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane + output_size_per_lane) * self.precision
                    # 전체 매트릭스에 필요한 타일 수 예측 (너무 작은 타일은 불리함)
                    n_tile = math.ceil(M / max(tile_M, 128)) * math.ceil(N / max(tile_N, 128))
                    check_spad_size = (used_spad_size < max_spad_size and used_spad_size_per_lane < max_spad_per_lane)
                    # 다양한 기준을 결합해 우수 후보 선정: 스패드 사용량, weight reuse, 최소 타일 수 등
                    if check_spad_size and max_used_spad_size < used_spad_size and maximize_i_j <= tile_M * tile_N and n_tile >= minimum_n_tile and max(tile_N, 128) // max(tile_M, 128) < 10:
                        max_used_spad_size = used_spad_size
                        maximize_i_j = tile_M * tile_N
                        mapping = (tile_M, tile_N, tile_K)
                    if check_spad_size:
                        tile_candidates.append((used_spad_size, (tile_M, tile_N, tile_K)))

        # 사용량 기준으로 후보 정렬 및 반환
        tile_candidates = sorted(tile_candidates, key=lambda x: x[0], reverse=True)
        tile_candidates = [v for _, v in tile_candidates]
        return tile_candidates

    def conv_combination_mapping(self, M, N, K, K_H, K_W, O_H, O_W, stride, dilation, n_extra_node=0):
        """컨볼루션을 GEMM으로 근사하여 타일 후보를 생성합니다.

        변수 설명: K_H/K_W 필터 차원, O_H/O_W 출력 차원, stride/dilation 등의 파라미터를 고려합니다.
        """
        tile_candidates = []
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2 # double buffer 고려
        max_spad_per_lane = spad_size_per_lane // 2 # lane 당 최대

        # 후보 선정용 보조 변수
        max_used_spad_size = 0
        # 먼저 GEMM 근사 값으로 M,N,K를 구합니다 (conv->GEMM 변환 관점)
        M, N, K = self.gemm_combination_mapping(M, N, K, n_extra_node=n_extra_node, pad_k=False, is_conv=True)[0]
        max_k_h_w = 1 # kernel size 최대화 보조
        max_o_h_w = 1 # output size 최대화 보조
        K = min(K, self.vector_lane)  # K는 vector lane 이하로 제한
        for o_h in sympy.divisors(O_H):
            for o_w in sympy.divisors(O_W):
                for k_h in sympy.divisors(K_H):
                    for k_w in sympy.divisors(K_W):
                        # 입력(ih,iw) 크기 계산: output/stride/dilation 고려
                        i_h = 1 + (o_h - 1) * stride[0] + (k_h - 1) * dilation[0]
                        i_w = 1 + (o_w - 1) * stride[1] + (k_w - 1) * dilation[1]
                        # 가중치/입력/출력의 스패드 사용량 계산
                        weight_size = k_w * k_h * K * N
                        input_size = i_w * i_h * M * K
                        output_size = o_w * o_h * M * N
                        used_spad_size = (weight_size + input_size + output_size * (1 + n_extra_node)) * self.precision
                        weight_size_per_lane = self.get_spad_size_per_lane(k_w * k_h * K, N)
                        input_size_per_lane = self.get_spad_size_per_lane(i_w * i_h * M, K)
                        output_size_per_lane = self.get_spad_size_per_lane(o_w * o_h * M  * (1 + n_extra_node), N)
                        used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane + output_size_per_lane) * self.precision
                        # lane 및 전체 스패드 제한을 넘지 않는지 확인
                        check_spad_size = (used_spad_size < max_spad_size and used_spad_size_per_lane < max_spad_per_lane)
                        if check_spad_size:
                            tile_candidates.append((used_spad_size, (k_h, k_w, o_h, o_w, M, N, K)))
                            if max_used_spad_size < used_spad_size and max_k_h_w <= k_h * k_w and max_o_h_w <= o_h * o_w:
                                max_used_spad_size = used_spad_size
                                max_k_h_w = k_h * k_w
                                max_o_h_w = o_h * o_w
                                mapping = (k_h, k_w, o_h, o_w, M, N, K)
        if max_used_spad_size == 0:
            raise RuntimeError("Cannot find a valid mapping")

        tile_candidates = sorted(tile_candidates, key=lambda x: x[0], reverse=True)
        tile_candidates = [v for _, v in tile_candidates]
        return tile_candidates

    def conv_multi_tile_mapping(self, M, N, K, K_H, K_W, O_H, O_W, stride, dilation, n_extra_node=0):
        """Create convolution tiling candidates that allow multi-tile decomposition along kernel width.

        설명: conv->GEMM 근사를 사용하되 K_W와 같은 커널 폭을 고려해 다중 타일 전략을 생성합니다.
        필요성: 일부 conv 설정에서 단일 타일로 충분히 표현할 수 없을 때, 효과적인 multi-tile 분해를 찾기 위해 사용됩니다.
        """
        tile_candidates = []
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2
        max_spad_per_lane = spad_size_per_lane // 2

        max_used_spad_size = 0
        M, N, K = self.gemm_combination_mapping(M, N, K * K_W, n_extra_node=n_extra_node, pad_k=False, is_conv=True)[0]
        max_k_h_w = K_W
        for o_h in sympy.divisors(O_H):
            for o_w in sympy.divisors(O_W):
                for k_h in sympy.divisors(K_H):
                    i_h = 1 + (o_h - 1) * stride[0] + (k_h - 1) * dilation[0]
                    i_w = 1 + (o_w - 1) * stride[1] + (K_W - 1) * dilation[1]
                    weight_size = 1 * k_h * K * N
                    input_size = i_w * i_h * M * K
                    output_size = o_w * o_h * M * N
                    used_spad_size = (weight_size + input_size + output_size * (1 + n_extra_node)) * self.precision
                    weight_size_per_lane = self.get_spad_size_per_lane(1 * k_h * K, N)
                    input_size_per_lane = self.get_spad_size_per_lane(i_w * i_h * M, K)
                    output_size_per_lane = self.get_spad_size_per_lane(o_w * o_h * M  * (1 + n_extra_node), N)
                    used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane + output_size_per_lane) * self.precision
                    check_spad_size = (used_spad_size < max_spad_size and used_spad_size_per_lane < max_spad_per_lane)
                    if check_spad_size:
                        tile_candidates.append((used_spad_size, (k_h, K_W, o_h, o_w, M, N, K)))
                        if max_used_spad_size < used_spad_size and max_k_h_w <= k_h:
                            max_used_spad_size = used_spad_size
                            max_k_h_w = k_h
                            mapping = (k_h, K_W, o_h, o_w, M, N, K)
        if max_used_spad_size == 0:
            raise RuntimeError("Cannot find a valid mapping")
        tile_candidates = sorted(tile_candidates, key=lambda x: x[0], reverse=True)
        tile_candidates = [v for _, v in tile_candidates]
        return tile_candidates

    def conv_single_batch_mapping(self, M, N, K, K_H, K_W, O_H, O_W, stride, dilation, n_extra_node=0):
        """Create convolution tiling candidates targeting single-batch usage.

        설명: 입력 배치가 1인 경우에 맞춘 conv 타일 후보를 생성합니다. stride/dilation 및 filter 크기를 반영합니다.
        필요성: 단일 배치에서 메모리/스패드 활용을 최적화하고 성능을 높이기 위해 사용됩니다.
        """
        tile_candidates = []
        spad_size_per_lane = self.spad_info["spad_size"]
        spad_size = spad_size_per_lane * self.vector_lane
        max_spad_size = spad_size // 2
        max_spad_per_lane = spad_size_per_lane // 2

        max_used_spad_size = 0
        M, N, K = self.gemm_combination_mapping(O_W, N, K, n_extra_node=n_extra_node, pad_k=False, is_conv=True)[0]
        max_k_h_w = 1
        for o_h in sympy.divisors(O_H):
            for k_h in sympy.divisors(K_H):
                for k_w in sympy.divisors(K_W):
                    i_h = 1 + (o_h - 1) * stride[0] + (k_h - 1) * dilation[0]
                    i_w = 1 + (M - 1) * stride[1] + (k_w - 1) * dilation[1]
                    weight_size = k_w * k_h * K * N
                    input_size = i_w * i_h * k_w * K
                    output_size = M * o_h * N
                    used_spad_size = (weight_size + input_size + output_size * (1 + n_extra_node)) * self.precision
                    weight_size_per_lane = self.get_spad_size_per_lane(k_w * k_h * K, N)
                    input_size_per_lane = self.get_spad_size_per_lane(i_w * i_h * k_w, K)
                    output_size_per_lane = self.get_spad_size_per_lane(M * o_h  * (1 + n_extra_node), N)
                    used_spad_size_per_lane = (weight_size_per_lane + input_size_per_lane + output_size_per_lane) * self.precision
                    check_spad_size = (used_spad_size < max_spad_size and used_spad_size_per_lane < max_spad_per_lane)
                    if check_spad_size:
                        tile_candidates.append((used_spad_size, (k_h, k_w, o_h, M, M, N, K)))
                        if max_used_spad_size < used_spad_size and max_k_h_w <= k_h * k_w:
                            max_used_spad_size = used_spad_size
                            max_k_h_w = k_h * k_w
                            mapping = (k_h, k_w, o_h, M, M, N, K)
        if max_used_spad_size == 0:
            raise RuntimeError("Cannot find a valid mapping")
        tile_candidates = sorted(tile_candidates, key=lambda x: x[0], reverse=True)
        tile_candidates = [v for _, v in tile_candidates]
        return tile_candidates

    def meta_kernel(self):
        """Prepare and register metadata needed by the wrapper and external tooling.

        이 메서드는 wrapper 코드에 출력할 루프 정보와 인자 속성을 정리하여 등록합니다.
        목적: 생성된 커널 코드와 외부 툴(예: 검증/벤치마크)이 필요로 하는 메타정보를 제공하기 위함입니다.
        """
        wrapper = V.graph.wrapper_code
        kernel_arg_attributes = self.kernel_arg_attributes
        _, _, arg_attributes, _ = self.kernel_group.args.mlir_argdefs()
        if kernel_arg_attributes is not None:
            for name, attr in kernel_arg_attributes:
                for idx in range(len(arg_attributes)):
                    if arg_attributes[idx][0] == name:
                        arg_attributes[idx][1] = attr
        wrapper.add_import_once('\nprint(f\'Wrapper Codegen Path = {__file__}\')')
        # Dump loop and load/store information
        wrapper.add_import_once(f"loop_info = {self.loop_info}")
        wrapper.add_import_once(f"arg_attributes = {arg_attributes}")

    def call_kernel(self, kernel_name):
        """Generate and register the wrapper call to the compiled kernel.

        역할: wrapper에 커널 호출 코드를 생성하여 외부(파이썬 또는 래퍼)에서 해당 커널을 실행할 수 있게 합니다.
        왜 필요한가: 템플릿으로 생성된 커널을 실제 호출 코드와 연결하기 위해 필요합니다.
        """
        wrapper = V.graph.wrapper_code
        _, call_args, _, _ = self.kernel_group.args.mlir_argdefs()
        # generate the code to call this
        wrapper.generate_kernel_call(
            kernel_name if self.outer_func_name is None else self.outer_func_name + f"_{len(call_args)}",
            call_args, cuda=False)
    
    # node = schedule buffer
    def codegen_template_code(self, render, template_node, prologue_nodes, epilogue_nodes, tile_info):
        """Generate source code for a template given its render and surrounding prologue/epilogue nodes.

        이 함수는 주어진 템플릿(render)을 실행하여 부분 코드를 얻고, prologue/epilogue 노드를 코드화하며
        필요한 load/store/reduction 훅을 교체하여 통합된 소스 코드를 반환합니다.
        왜 필요한가: 템플릿 기반 커널의 전체 소스(프로로그/에필로그 포함)를 일관되게 생성하기 위해 필요합니다.
        """
        with self as kernel:
            _, _, _, kernel.buffer_types = self.kernel_group.args.mlir_argdefs()
            for node in [template_node, *prologue_nodes, *epilogue_nodes]:
                node.mark_run()

            # Partial codgen template nodes
            partial_code = render(kwargs={**render.keywords['kwargs'], 'tile_info': tile_info})

            # Swap load/store functions
            kernel.load = kernel.load_epilogue
            kernel.store = kernel.store_epilogue
            kernel.store_reduction = kernel.store_reduction_epilogue
            kernel.reduction = kernel.reduction_epilogue

            # Codegen prologue nodes
            if prologue_nodes:
                # Flush created varaibles, since template fusion doen't share variable
                with kernel.prologue_buffer_group.as_local():
                    _, (group, reduction_group) = max(
                        [prologue_nodes[-1]], key=lambda x: int(x.is_reduction())
                    ).group
                    prologue_tile_desc = kernel.set_tile_size(kernel.prologue_info, prologue=True)
                    kernel.kernel_group.set_tile_info(prologue_tile_desc)
                    vars, reduction_vars = kernel.set_ranges(group, reduction_group)
                    for node in prologue_nodes:
                        # Reuse created spad
                        read_list = sorted([i.name for i in node.read_writes.reads])
                        candidate_found = False
                        # Why? There is a case that memdep.get_size() != data.get_size()
                        buf_dict = {}
                        buf_dict.update({val.name : val for val in V.graph.buffers})
                        buf_dict.update(V.graph.graph_inputs)
                        for candidate_read in read_list:
                            if candidate_read in buf_dict and reduce(operator.mul, buf_dict[candidate_read].get_size(), 1) == node.node.get_numel():
                                prologue_input_arg = candidate_read
                                candidate_found = True
                                break
                        assert(candidate_found)
                        assert(len(node.read_writes.writes)==1)
                        prologue_output_arg = list(node.read_writes.writes)[0].name
                        template_buf = self.kernel_group.args.input_buffers[prologue_output_arg]
                        target_buf = f"{template_buf}_buffer" # FIXME. How to pass spad buffer name?

                        # To skip the dma code gen
                        kernel.buffer_names[prologue_input_arg] = target_buf
                        kernel.buffer_names[prologue_output_arg] = target_buf

                        # Edge delete
                        kernel.kernel_group.args.input_buffers = {
                            (arg if buf != template_buf else prologue_input_arg): buf
                            for arg, buf in kernel.kernel_group.args.input_buffers.items()
                        }
                        node.codegen((vars, reduction_vars))

            # Codegen epilogue nodes
            tile_desc = kernel.set_tile_size(kernel.epilogue_info)
            kernel.kernel_group.set_tile_info(tile_desc)
            kernel.call_ranges = None
            if epilogue_nodes:
                with kernel.epilogue_buffer_group.as_local():
                    _, (group, reduction_group) = max(
                        epilogue_nodes, key=lambda x: int(x.is_reduction())
                    ).group
                    vars, reduction_vars = kernel.set_ranges(group, reduction_group)
                    for node in epilogue_nodes:
                        node.codegen((vars, reduction_vars))

        with V.set_kernel_handler(kernel):
            src_code = (
                partial_code
                if isinstance(partial_code, str)
                else partial_code.finalize()
            )

            # For consistency, white space could make wrong write_path
            buffer = IndentedBuffer()
            buffer.splice(src_code)
            src_code = buffer.getvalue()
            self._prepare_simulator_headers(src_code)
        return src_code

    def make_choices(self, tile_candidates, render, template_node, prologue_nodes, epilogue_nodes):
        """For each tile candidate, generate code, run benchmark and collect results.

        목적: 자동 튜닝을 위해 후보별로 코드를 생성하고 실행(벤치마크) 결과를 수집하여 최적안을 찾을 수 있게 합니다.
        """
        choices = []
        for tile_info in tile_candidates:
            if extension_config.CONFIG_DEBUG_MODE:
                # Compute Tile M, N, K DMA Tile M, N, K
                print(f"[Auto-tune] Trying tile size: {list(tile_info)}")
            src_code = self.codegen_template_code(render, template_node, prologue_nodes, epilogue_nodes, tile_info)
            bench_runner = self.run_bench([template_node], self.kernel_name, src_code)
            choices.append((bench_runner, src_code, tile_info, self.loop_size))
            self.reset(reason=None)
        return choices

    def _log_autotune_result(self, best_choice, best_cycle):
        """Log the result of autotuning (best tile size and cycles).

        필요성: 자동 튜닝 결과를 사용자에게 알려주고 디버깅/분석에 사용됩니다.
        """
        tile_size = best_choice[2]
        print(
            f"[Auto-tune] Optimal tile size: {list(tile_size)}, "
            f"cycles: {best_cycle}"
        )

    def codegen_nodes(self, tile_candidates, render, template_node, prologue_nodes, epilogue_nodes):
        """Top-level API to produce source for given template nodes.

        동작: autotune 설정에 따라 자동 튜닝을 실행하거나(있다면), 첫 후보 또는 단일 타일로 코드를 생성합니다.
        왜 필요한가: 실제 커널 소스 생성의 진입점으로 상위 로직이 이 함수를 호출합니다.
        """
        if "autotune" in extension_config.codegen_mapping_strategy and len(tile_candidates):
            src_code, loop_size = self.autotune(tile_candidates, render, template_node, prologue_nodes, epilogue_nodes)
            self.loop_size = loop_size
        else:
            tile_info = tile_candidates[0] if tile_candidates else None
            src_code = self.codegen_template_code(render, template_node, prologue_nodes, epilogue_nodes, tile_info)

        with V.set_kernel_handler(self):
            self.meta_kernel()
        return src_code

    def _prepare_simulator_headers(self, src_code):
        spad_end_symbol = f"int spad_end[0] __attribute__ ((section(\".spad\")));\n"
        spad_section_end_symbol = f"int spad_section_end[0] __attribute__ ((section(\".spad\"), aligned({self.spad_info['spad_size']*self.vector_lane})));"

        write_path = extension_codecache.get_write_path(src_code)
        if not os.path.exists(write_path):
            os.makedirs(write_path, exist_ok=True)
        spike_write_path = os.path.join(write_path, "global_var.h")
        gem5_write_path = os.path.join(write_path, "gem5_global_var.h")
        if not os.path.exists(spike_write_path):
            write_atomic(spike_write_path, self.header.getvalue()+spad_end_symbol+spad_section_end_symbol)
        if not os.path.exists(gem5_write_path):
            write_atomic(gem5_write_path, self.gem5_header.getvalue())

    def codegen_prologue_body(self):
        """Generate the prologue portion of the kernel body (DMA loads, spad setup, prologue compute).

        왜 필요한가: prologue는 타일의 입력/가중치 로드와 초기화 작업을 수행하며, main compute 이전에 필요한 준비 코드를 제공합니다.
        """
        body = IndentedBuffer()
        with self.prologue_buffer_group.as_local():
            body.splice(self.spad_buffer)
            body.splice(self.applys)
            body.splice(self.dma_loads)

            if (self.loads.getvalue() != '' or self.compute.getvalue() != '' or self.stores.getvalue() != ''):
                body.writelines(self.prologue_compute_body_loop.lines())
                compute_body = mlir_common.ParallelLoopBuffer()
                with contextlib.ExitStack() as stack:
                    stack.enter_context(compute_body.indent(attribute="{inner_loop=false}"))
                    compute_body.splice(self.loads)
                    compute_body.splice(self.compute)
                    compute_body.splice(self.stores)
                body.splice(compute_body)
            body.splice(self.dma_stores)
        return body

    def codegen_epilogue_body(self):
        """Generate the epilogue portion of the kernel body (stores, reduction handling, DMA outs).

        목적: 메인 계산 후 출력 저장과 리덕션 처리 등 후처리를 관리하여 결과를 메모리로 내보내는 역할을 합니다.
        """
        def template_store():
            dram_var = self.epilogue_info["dram_var"]
            index_list = self.epilogue_info["dram_idx"]
            tile_desc = self.epilogue_info["dram_tile_desc"]
            code = self.def_dma_op("MVOUT", dram_var, index_list, tile_desc)
            self.cse.generate(self.dma_stores, code, assignment = False)

        body = IndentedBuffer()
        with self.epilogue_buffer_group.as_local():
            # Do dma store first to overlap epilogue nodes
            if self.reduction_fusion:
                if len(self.stores._lines) == 0:
                    template_store()
                    body.splice(self.dma_stores)
                    self.dma_stores.clear()
            body.splice(self.spad_buffer)
            body.splice(self.applys)
            body.splice(self.dma_loads)
            body.writelines(self.compute_body_loop.lines())
            compute_body = mlir_common.ParallelLoopBuffer()
            with contextlib.ExitStack() as stack:
                stack.enter_context(compute_body.indent(attribute="{inner_loop=false}",suffix=self.compute_body_loop.epilogue_line()))
                if self.reduction_fusion:
                    compute_body.writelines(self.reduction_body_loop.lines())
                    compute_body.splice(self.masks)
                    stack.enter_context(compute_body.indent(attribute="{inner_loop=false}"))
                    compute_body.splice(self.loads)
                    compute_body.splice(self.compute)
                else:
                    compute_body.splice(self.loads)
                    compute_body.splice(self.compute)
                    if len(self.stores._lines) == 0:
                        template_store()
                compute_body.splice(self.stores)
            if (compute_body.getvalue()):
                body.splice(compute_body)
            body.splice(self.dma_stores)
            body.splice(self.reduction_epilogue_suffix)
        return body

    def def_kernel(
        self,
        inputs: List[IRNode],
        outputs: List[IRNode],
        names_str: str = "",
        input_reorder: Optional[List[int]] = None,
    ) -> str:
        """Register kernel input/output names and hook to render function signature.

        역할: 입력/출력 노드와 이름을 매핑하고, 렌더 시 사용할 인자 정의 훅을 등록합니다.
        왜 필요한가: 템플릿이 생성한 커널을 외부에서 호출할 때 정확한 인자 시그니처를 제공하기 위해 필요합니다.
        """
        names = [x.strip() for x in names_str.strip().split(",")]
        if len(inputs) + len(outputs) != len(names):
            raise RuntimeError(
                f"{len(inputs) + len(outputs)=} != {len(names)=}, {inputs=}, {outputs=}, {names=}"
            )

        if input_reorder is not None:
            assert len(inputs) == len(input_reorder)
        else:
            input_reorder = list(range(len(inputs)))

        for idx in input_reorder:
            name = names[idx]
            node = inputs[idx]
            if node is not None:
                self.named_nodes[name] = node
                self.kernel_group.args.input_buffers[node.get_name()] = name

        extra_node = {}
        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            if node is not None:
                self.named_nodes[name] = node
                self.kernel_group.args.output_buffers[node.get_name()] = name
                self.store_buffer_names.add(node.get_name())    #TODO: Is this enough not calling store() in mlir_common.py?
                if isinstance(node, SchedulerNode):
                    extra_node[node.get_name()] = node.node
                else:
                    extra_node[node.get_name()] = node
                self.buffer_names[node.get_name()] = self.epilogue_info['sram_var']

        def hook():
            arg_defs, *_ = self.kernel_group.args.mlir_argdefs(extra_node=extra_node)
            return f"({', '.join(arg_defs)})"

        assert "<DEF_KERNEL>" not in self.render_hooks
        self.render_hooks["<DEF_KERNEL>"] = hook
        return "<DEF_KERNEL>"

    # This function is a temporal function for convolution because currently convolution kernel is not considering padding.
    # Padding is done by python wrapper so the padded input size is manually applied here.
    def def_conv_kernel(
        self,
        inputs: List[IRNode],
        outputs: List[IRNode],
        names_str: str = "",
        padded_input_size: List[int] = [],
        input_reorder: Optional[List[int]] = None,
    ) -> str:
        """Define convolution-specific kernel signature and handle padded input size adjustments.

        이유: convolution의 경우 파이썬 래퍼에서 패딩을 처리하므로 템플릿 시그니처에 패딩된 입력 크기를 반영해야 합니다.
        """
        names = [x.strip() for x in names_str.strip().split(",")]
        if len(inputs) + len(outputs) != len(names):
            raise RuntimeError(
                f"{len(inputs) + len(outputs)=} != {len(names)=}, {inputs=}, {outputs=}, {names=}"
            )

        if input_reorder is not None:
            assert len(inputs) == len(input_reorder)
        else:
            input_reorder = list(range(len(inputs)))

        for idx in input_reorder:
            name = names[idx]
            node = inputs[idx]
            if node is not None:
                self.named_nodes[name] = node
                self.kernel_group.args.input_buffers[node.get_name()] = name

        self.extra_node = {}
        for name, node in zip(names[len(inputs) : len(inputs) + len(outputs)], outputs):
            if node is not None:
                self.named_nodes[name] = node
                self.kernel_group.args.output_buffers[node.get_name()] = name
                self.store_buffer_names.add(node.get_name())    #TODO: Is this enough not calling store() in mlir_common.py?
                self.extra_node[node.get_name()] = node
                self.buffer_names[node.get_name()] = self.epilogue_info['sram_var']   #TODO: Buffer name fixed

        def kernel_hook():
            arg_defs, *_ = self.kernel_group.args.mlir_argdefs(extra_node=self.extra_node)
            arg_defs[0] = re.sub(r'(\d+)(?=xf32)', str(padded_input_size), arg_defs[0])
            return f"({', '.join(arg_defs)})"

        assert "<DEF_CONV_KERNEL>" not in self.render_hooks
        self.render_hooks["<DEF_CONV_KERNEL>"] = kernel_hook
        return "<DEF_CONV_KERNEL>"

    # This function is for convolution wrapper function finalizing.
    def def_wrapper(self, only_store_buffer: bool = False, epilogue_buffer: str = False):
        """Register a wrapper function signature hook used to finalize convolution wrappers.

        목적: 파이썬 레벨의 래퍼 함수에서 사용할 인자 시그니처를 정의합니다(주로 buffer 이름만 전달).
        """
        def wrapper_hook():
            arg_defs, *_ = self.kernel_group.args.mlir_argdefs(extra_node=self.extra_node)
            wrapper_arg_defs = [arg.split('%')[1].split(':')[0] for arg in arg_defs]
            return f"({', '.join(wrapper_arg_defs)})"

        if "<DEF_CONV_WRAPPER>" not in self.render_hooks:
            self.render_hooks["<DEF_CONV_WRAPPER>"] = wrapper_hook
        return "<DEF_CONV_WRAPPER>"

    def get_conv_inputs(self):
        """Return mapping of convolution input buffer names used by the kernel.

        유용성: convolution wrapper/외부 코드가 입력 버퍼 이름을 필요로 할 때 호출됩니다.
        """
        return self.kernel_group.args.input_buffers

    def get_conv_outputs(self):
        """Return mapping of convolution output buffer names that are actively used (not REMOVED).

        유용성: wrapper가 출력 버퍼들을 쿼리할 때 사용됩니다.
        """
        return {k: v for k, v in self.kernel_group.args.output_buffers.items() if v != 'REMOVED'}

    def load_input(self, indent_size: int = 0):
        """Create a render hook that prepares input (DMA-in and prologue) code for the kernel.

        이유: 입력 데이터와 가중치를 타일까지 맞추어 DRAM에서 SRAM/SPAD로 불러오는 코드를 생성합니다.
        """
        def hook():
            code = IndentedBuffer()
            prologue_code = self.codegen_prologue_body()
            if prologue_code.getvalue():
                input_dma_code = self.def_dma_op("MVIN", self.prologue_info["input_dram_var"], self.prologue_info["input_idx"],
                                self.prologue_info["input_tile_desc"], subtile_size=self.prologue_info["input_subtile_size"], async_type=False)
                weight_dma_code = self.def_dma_op("MVIN", self.prologue_info["weight_dram_var"], self.prologue_info["weight_idx"],
                                self.prologue_info["weight_tile_desc"], subtile_size=self.prologue_info["weight_subtile_size"], async_type=False)
                if (self.prologue_info["is_input_fused"]):
                    code.splice(input_dma_code)
                    code.splice(prologue_code)
                    code.splice(weight_dma_code)
                else:
                    code.splice(weight_dma_code)
                    code.splice(prologue_code)
                    code.splice(input_dma_code)
            else:
                dma_code = self.def_dma_op("MVIN", self.prologue_info["input_dram_var"], self.prologue_info["input_idx"],
                                self.prologue_info["input_tile_desc"], subtile_size=self.prologue_info["input_subtile_size"], async_type=False)
                code.splice(dma_code)
                dma_code = self.def_dma_op("MVIN", self.prologue_info["weight_dram_var"], self.prologue_info["weight_idx"],
                                self.prologue_info["weight_tile_desc"], subtile_size=self.prologue_info["weight_subtile_size"], async_type=False)
                code.splice(dma_code)
            code = textwrap.indent(code.getvalue(), " "*indent_size).strip()
            return code

        assert "<PREPARE_INPUT>" not in self.render_hooks
        self.render_hooks["<PREPARE_INPUT>"] = hook
        self.render_hooks.move_to_end("<PREPARE_INPUT>", last=False) # Force order to be triggered first
        return "<PREPARE_INPUT>"

    def store_output(self, indent_size: int = 0):
        """Register a render hook that returns the epilogue (store/output) code.

        목적: 커널의 출력 저장/후처리 코드를 템플릿 렌더링 과정에서 올바른 위치에 삽입하기 위해 필요합니다.
        """
        def hook():
            epilogue_code = self.codegen_epilogue_body()
            return textwrap.indent(epilogue_code.getvalue(), " "*indent_size).strip()

        assert "<STORE_OUTPUT>" not in self.render_hooks
        self.render_hooks["<STORE_OUTPUT>"] = hook
        self.render_hooks.move_to_end("<STORE_OUTPUT>", last=False) # Force order to be triggered first
        return "<STORE_OUTPUT>"

    def reduction_output(self, indent_size: int = 0):
        """Register a hook that injects reduction-specific output code into rendered template.

        이유: 리덕션 연산의 특수한 후처리 코드(축소 결과 집계 등)를 템플릿의 출력 부분에 주입하기 위해 사용됩니다.
        """
        def hook():
            return textwrap.indent(self.reductions_suffix.getvalue(), " "*indent_size).strip()

        assert "<REDUCTION_OUTPUT>" not in self.render_hooks
        self.render_hooks["<REDUCTION_OUTPUT>"] = hook
        return "<REDUCTION_OUTPUT>"

    def def_function(self):
        """Optionally define an outer (Python) function wrapper for the kernel.

        목적: 외부에서 호출 가능한 파이썬 래퍼를 생성하거나, 없다면 None을 반환합니다.
        """
        _, call_args, _ = self.kernel_group.args.python_argdefs()
        if self.outer_func_render is not None:
            partial_code, function_name = self.outer_func_render(input_args=call_args)
            return PartialRender(
                partial_code,
                self.render_hooks,
            ), function_name
        else:
            return None, None

    def def_global_vars(self):
        """Register global variable definitions hook for the template rendering.

        이유: 템플릿에서 필요한 전역 변수(예: 헤더에 들어갈 상수)를 렌더 시 삽입하기 위해 사용됩니다.
        """
        key = "<GLOBAL_VARS>"
        def hook():
            return textwrap.indent(self.global_vars.getvalue(), "").strip()

        assert key not in self.render_hooks
        self.render_hooks[key] = hook
        return key

    def def_local_vars(self, indent_size=0):
        """Register local variable definitions (constants and allocations) for rendering.

        목적: 커널 내부에서 사용되는 상수/할당 변수를 정의하고 템플릿 내부에서 참조 가능하게 합니다.
        """
        key = "<LOCAL_VARS>"
        def hook():
            code = IndentedBuffer()
            code.tabwidth = 1
            code.splice(self.const_buffer)
            code.splice(self.alloc_buffer)
            return textwrap.indent(code.getvalue(), " "*indent_size).strip()

        assert key not in self.render_hooks
        self.render_hooks[key] = hook
        return key

    def def_dma_op(self, dma_type, dram_var:str, index_list:list, tile_desc:mlir_common.MLIRMultiDimTile,
                   subtile_size:list=[], async_type=None, indent_size=0):
        """Generate DMA operation code (MVIN/MVOUT) for given DRAM variable and tile descriptor.

        필요성: DRAM <-> SPAD(SRAM) 이동을 MLIR/시뮬레이터용 코드로 변환하기 위해 사용됩니다. subtile/async 옵션을 통해 세부 동작을 제어합니다.
        """
        # Prepare code block
        local_code = IndentedBuffer()
        with V.set_kernel_handler(self):
            index_var = self.parse_index_list(index_list, local_code, offset=tile_desc.offset)
            node_layout = self.named_nodes[dram_var].get_layout()
            if dram_var in self.exception_nodes:
                numel = self.exception_nodes[dram_var]["numel"]
            else:
                numel = self.get_arg_info(self.named_nodes[dram_var].get_name()).get_numel()
            mlir_dtype = mlir_common.DTYPE_TO_MLIR[node_layout.dtype]
            dram_shape = f"memref<{numel}x{mlir_dtype}>"
            dram_stride = []
            for idx in index_list:
                if idx.is_Mul:
                    dram_stride.append(int(idx.args[0]))
                elif idx == sympy.Symbol("c0"):
                    dram_stride.append(0)
                elif not idx.is_Number:
                    dram_stride.append(1)
                else:
                    dram_stride.append(0)

            sram_var = tile_desc.get_name()
            tile_shape = tile_desc.get_mlir_shape(mlir_dtype)
            tile_stride = tile_desc.get_tile_stride()
            vlane_split_axis = tile_desc.vmap.vlane_split_axis
            vlane_stride = tile_desc.vmap.vlane_stride

            zero_cse = self.get_const_cse(0, "index")
            sram_index_var = ", ".join([f"%{str(zero_cse)}"]*tile_desc.get_nr_dim())

            attribute_parts = [f"dram_stride={dram_stride}", f"sram_stride={tile_stride}", "padding=0"]
            if subtile_size:
                attribute_parts.append(f"subtile_size={subtile_size}, async={int(async_type) if async_type is not None else 1}")
            attribute = "  {" + ", ".join(attribute_parts) + "}"
            code = self.get_dma_code(dma_type, vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                     dram_shape, tile_shape, "")
            local_code.writeline(code)
            local_code.writeline(attribute)
        return textwrap.indent(local_code.getvalue(), " "*indent_size).strip()

    def def_sram_buffer(self, dram_name, tile_desc, id=0, indent_size=0):
        """Define/get the SRAM (SPAD) buffer memref declaration for a given DRAM name and tile.

        목적: 타일용 SRAM 전역 버퍼를 할당하고 해당 global memref 선언 코드를 반환합니다.
        """
        # Prepare code block
        with V.set_kernel_handler(self):
            dtype = self.named_nodes[dram_name].get_layout().dtype
            tile_shape = tile_desc.get_mlir_shape(mlir_common.DTYPE_TO_MLIR[dtype])
            buffer_name = self.allocate_sram_buffer(dtype, dram_name, tile_desc, id, forced_name=dram_name)
            code = f"%{tile_desc.name} = memref.get_global @{buffer_name} : {tile_shape}"
        return textwrap.indent(code, " "*indent_size).strip()

    def render(self, template, kwargs, define_function=None):
        """Render an MLIR template and attach rendering hooks.

        역할: 주어진 템플릿을 실제 코드 문자열로 렌더링하고, 필요한 경우 define_function을 통해 훅을 등록합니다.
        """
        code = template.render(**kwargs)
        if define_function is not None:
            define_function(self)

        return PartialRender(
            code,
            self.render_hooks,
        )

    def get_spad_size_per_lane(self, tile_m, tile_n):
        """Estimate SPAD usage per lane given tile dimensions.

        사용 이유: SPAD 사용량을 타일링/매핑 후보 평가에서 비교하기 위함입니다.
        """
        size = tile_m * ((tile_n + self.vector_lane - 1) // self.vector_lane)
        return max(size, 2) # vector load/store

    def load_epilogue(self, name: str, index: sympy.Expr):
        """Load data from SRAM (epilogue path) into vector registers for computation.

        목적: epilogue 모드에서 SRAM에서 벡터를 읽어오는 코드를 생성하여 리덕션/후처리 계산에 사용됩니다.
        """
        index = self.rename_indexing(index)
        dram_var = self.kernel_group.args.input(name)
        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]

        # Want to use tile_desc from epilogue_info
        index_var = self.parse_indices(index)
        dram_stride = [index.coeff(sympy.Symbol(val)) for val in self.dim_aliasing.values()]
        vlane_split_axis = self.kernel_group.tile_desc.vmap.vlane_split_axis
        vlane_stride = self.kernel_group.tile_desc.vmap.vlane_stride
        tile_shape = self.kernel_group.tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = self.kernel_group.tile_desc.get_tile_stride()

        # Compute vector unit size
        vshape = self.kernel_group.tile_desc.get_mlir_vshape(mlir_dtype)
        compute_vec_size = self.kernel_group.tile_desc.get_compute_vec_size()

        if name not in self.buffer_names:
            # Allocate sram buffer
            dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
            sram_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, self.kernel_group.tile_desc, index)
            attribute = f"{{dram_stride={dram_stride}, sram_stride={tile_stride}, padding=0}}"
            code = self.get_dma_code("MVIN", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                     dram_shape, tile_shape, attribute)
            self.cse.generate(self.dma_loads, code, assignment = False)
            self.buffer_names[name] = sram_var
        else:
            sram_var = self.buffer_names[name]

        # Load vector from sram
        zero_var = self.get_const_cse(0)
        if not self.reduction_fusion:
            compute_index_var = ",".join([f"%{zero_var}"] * (self.kernel_group.tile_desc.get_nr_dim()-1) + [f"%{self.compute_idx}"])
            if compute_vec_size > 1:
                operation = "affine.vector_load"
                line = f"{operation} %{sram_var}[{compute_index_var}] : {tile_shape}, {vshape}"
            else:
                operation = "affine.load"
                line = f"{operation} %{sram_var}[{compute_index_var}] : {tile_shape}"
            out = self.cse.generate(self.loads, line)
            self.register_var_info(out, [compute_vec_size, mlir_dtype])
        else: # For reduction case
            reduce_size = self.reduction_nr_outer_loop
            vsize = compute_vec_size//reduce_size
            vshape = f"vector<{vsize}x{mlir_dtype}>"

            if compute_vec_size > 1:
                offset = self.cse.generate(self.loads, f"affine.apply affine_map<(d0, d1) -> (d0 + d1*{(self.r_tile_size)})>(%{self.compute_idx}, %{self.reduction_loop_idx})")
                compute_index_var = ",".join([f"%{zero_var}"] * (self.kernel_group.tile_desc.get_nr_dim()-1) + [f"%{offset}"])
                operation = "affine.vector_load"
                line = f"{operation} %{sram_var}[{compute_index_var}] : {tile_shape}, {vshape}"
                out = self.cse.generate(self.loads, line)
            else:
                line = f"{operation} %{sram_var}[{compute_index_var}] : {tile_shape}"
                out = self.cse.generate(self.loads, line)
            self.register_var_info(out, [self.compute_body_loop.step, mlir_dtype])
        return out

    def store_epilogue(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        """Store a computed value back into SRAM and schedule DMA out if necessary (epilogue path).

        필요성: epilogue에서 계산된 값을 SRAM에 저장하고 최종적으로 DRAM으로 MVOUT을 생성하여 결과를 내보냅니다.
        """
        index = self.rename_indexing(index)
        dram_var = self.kernel_group.args.output(name)
        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]

        index_var = self.parse_indices(index)
        dram_stride = [index.coeff(sympy.Symbol(val)) for val in self.dim_aliasing.values()]
        vlane_split_axis = self.kernel_group.tile_desc.vmap.vlane_split_axis
        vlane_stride = self.kernel_group.tile_desc.vmap.vlane_stride
        tile_shape = self.kernel_group.tile_desc.get_mlir_shape(mlir_dtype)
        tile_stride = self.kernel_group.tile_desc.get_tile_stride()

        # Compute vector unit size
        vshape = self.kernel_group.tile_desc.get_mlir_vshape(mlir_dtype)
        compute_vec_size = self.kernel_group.tile_desc.get_compute_vec_size()

        if name not in self.buffer_names:
            sram_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, self.kernel_group.tile_desc, index)
            self.buffer_names[name] = sram_var
            store_force = False
        else:
            zero_cse = self.get_const_cse(0)
            sram_dims = len(tile_shape.split("x")) - 1
            sram_index_var = ",".join([f"%{zero_cse}"] * sram_dims)
            store_force = True
        sram_var = self.buffer_names[name]
        zero_var = self.get_const_cse(0)

        _, operand_type = self.var_info[value]
        if mlir_dtype != operand_type:
            value = ops.to_dtype(value, mlir_dtype, var_info=self.var_info)
        compute_index_var = ",".join([f"%{zero_var}"] * (self.kernel_group.tile_desc.get_nr_dim()-1) + [f"%{self.compute_idx}"])
        # Generate vector load instruction
        if compute_vec_size > 1:
            operation = "affine.vector_store"
            line = f"{operation} %{value}, %{sram_var}[{compute_index_var}] : {tile_shape}, {vshape}"
        else:
            operation = "affine.store"
            line = f"{operation} %{value}, %{sram_var}[{compute_index_var}] : {tile_shape}"
        line = line if store_force else DeferredLine(name, line)
        self.stores.writeline(line)

        # Generate DMA instruction
        attribute = f"{{dram_stride={dram_stride}, sram_stride={tile_stride}, padding=0}}"
        code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                 dram_shape, tile_shape, attribute)
        self.dma_stores.writeline(DeferredLine(name, code))

    def reduction_epilogue(self, dtype, src_dtype, reduction_type, value):
        """Handle generation of partial reduction storage and merging logic for a reduction operation.

        필요성: 리덕션의 중간 결과를 로드/결합/저장하여 최종 결과를 생성하는데 필요한 코드를 생성합니다. Welford 등 특별한 리덕션도 처리합니다.
        """
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        if argmax_or_argmin:
            raise NotImplementedError() #TODO: argmin, argmax
        if is_welford_reduction(reduction_type):
            if reduction_type == "welford_combine":
                raise NotImplementedError("welford_combine")
            else:
                assert reduction_type == "welford_reduce"
                type_name = mlir_common.DTYPE_TO_MLIR[dtype]
                reduction_key = src_dtype, reduction_type, value
                sum = self.reduction_epilogue(dtype, src_dtype, "sum", value)
                sqr_sum = self.reduction_epilogue(dtype, src_dtype, "sum", ops.mul(value, value))
                self.welford_reduce_out = (sum, sqr_sum, None)
                return sum, sqr_sum, None

        # Check duplicated reductions
        reduction_key = src_dtype, reduction_type, value
        if reduction_key in self.reduction_epilogue_result:
            return self.reduction_epilogue_result[reduction_key]

        # Reduction fusion codegen part
        vec_size = self.compute_body_loop.step
        type_name = mlir_common.DTYPE_TO_MLIR[dtype]
        new_tile_size = self.kernel_group.tile_desc.get_tile_size()[:-1] + [vec_size]
        new_vlane_split_axis = self.kernel_group.tile_desc.vmap.vlane_split_axis
        new_vlane_stride = self.kernel_group.tile_desc.vmap.vlane_stride
        local_tile_desc = mlir_common.MLIRMultiDimTile(new_tile_size, self.vector_lane, new_vlane_split_axis, new_vlane_stride, vec_size)

        tile_shape = local_tile_desc.get_mlir_shape(type_name)
        vshape = local_tile_desc.get_mlir_vshape(type_name)

        name = f"{reduction_type}_buffer{self.reduction_buffer_idx}"
        self.reduction_buffer_idx += 1
        index = "dummy_index" # Not used
        sram_var, _ = self.get_scratchpad_buffer(dtype, name, local_tile_desc, index, self.const_buffer)
        self.reduction_epilogue_result[reduction_key] = sram_var

        # Load partial result
        zero_var_list = [f"%{self.get_const_cse(0)}"] * local_tile_desc.get_nr_dim()
        zero_var_list[-2] = f"%{self.reduction_loop_idx}"
        compute_index_var = ", ".join(zero_var_list)
        operation = "affine.vector_load"
        line = f"{operation} %{sram_var}[{compute_index_var}] : {tile_shape}, {vshape}"
        out = self.cse.generate(self.loads, line)
        self.register_var_info(out, [self.compute_body_loop.step, type_name])

        # Reduction body codegen
        init = self.const_cse.generate(self.const_buffer, f"arith.constant {reduction_init(reduction_type, dtype)} : {type_name}")
        init_vec = self.const_cse.generate(self.const_buffer, f"vector.broadcast %{init} : {type_name} to {vshape}")
        self.register_var_info(init_vec, [local_tile_desc.get_compute_vec_size(), type_name])
        mask_shape, mask_var = self.get_mask()
        if mask_var is not None:
            value = ops.where(mask_var, value, init_vec)
        result = reduction_partial_combine_vec(reduction_type, value, out)

        # Store partial result
        operation = "affine.vector_store"
        line = f"{operation} %{result}, %{sram_var}[{compute_index_var}] : {tile_shape}, {vshape}"
        self.compute.writeline(line) # Need to be placed after partial reduction
        self.reduction_info[sram_var] = [reduction_type, local_tile_desc]
        return sram_var

    def store_reduction_epilogue(self, name, index, value):
        """Finalize the reduction by combining partial results and emitting MVOUT to DRAM.

        필요성: 여러 단계로 나뉜 리덕션의 파셜 결과들을 합치고, 최종적으로 DRAM에 저장하는 절차를 담당합니다.
        """
        index = self.rename_indexing(index)
        dram_var = self.kernel_group.args.output(name)
        dram_shape = mlir_common.MLIRKernelArgs.get_mlir_shape(self.buffer_types[name])
        dtype = V.graph.get_dtype(name)
        mlir_dtype = mlir_common.DTYPE_TO_MLIR[dtype]

        index_var = self.parse_indices(index, self.reductions_suffix, comments="// Store reduction")
        dram_stride = [index.coeff(sympy.Symbol(val)) for val in self.dim_aliasing.values()][:-1] # Assume that there is only one reduction axis
        vlane_split_axis = self.kernel_group.tile_desc.vmap.vlane_split_axis
        vlane_stride = self.kernel_group.tile_desc.vmap.vlane_stride

        # Create final buffer descriptor
        nr_outer_loop = self.reduction_nr_outer_loop
        tile_size = self.kernel_group.tile_desc.get_tile_size()[:-1]
        final_tile_desc = mlir_common.MLIRMultiDimTile(tile_size, self.vector_lane, vlane_split_axis, vlane_stride*nr_outer_loop*2)
        final_tile_shape = final_tile_desc.get_mlir_shape(mlir_dtype)
        final_tile_stride = final_tile_desc.get_tile_stride()
        sram_var, sram_index_var = self.get_scratchpad_buffer(dtype, name, final_tile_desc, index, buffer=self.const_buffer)

        # Set partial buffer descriptor
        partial_tile_desc = self.reduction_info[value][1]
        partial_vec_size = partial_tile_desc.get_compute_vec_size()
        partial_vshape = partial_tile_desc.get_mlir_vshape(mlir_dtype)
        partial_tile_shape = partial_tile_desc.get_mlir_shape(mlir_dtype)

        # Prepare constant
        init = self.const_cse.generate(self.const_buffer, f"arith.constant {reduction_init(self.reduction_info[value][0], dtype)} : {mlir_dtype}")
        partial_zero_var_list = [f"%{self.get_const_cse(0)}"] * partial_tile_desc.get_nr_dim()
        final_zero_var_list = [f"%{self.get_const_cse(0)}"] * final_tile_desc.get_nr_dim()
        for i in range(self.reduction_body_loop.size):
            # Load partial result
            body_index_var = self.const_cse.generate(self.const_buffer, f"arith.constant {i} : index")
            partial_zero_var_list[-2] = f"%{body_index_var}"
            compute_index_var = ",".join(partial_zero_var_list)

            operation = "affine.vector_load"
            line = f"{operation} %{value}[{compute_index_var}] : {partial_tile_shape}, {partial_vshape}"
            out = self.cse.generate(self.reductions_suffix, line)
            operation = "affine.vector_store"
            init_vec = self.const_cse.generate(self.const_buffer, f"vector.broadcast %{init} : {mlir_dtype} to {partial_vshape}")
            line = f"{operation} %{init_vec}, %{value}[{compute_index_var}] : {partial_tile_shape}, {partial_vshape}"
            self.reductions_suffix.writeline(line)

        # MVOUT Encoding
        # Generate DMA instruction
        attribute = f"{{dram_stride={dram_stride}, sram_stride={final_tile_stride}, padding=0}}"
        code = self.get_dma_code("MVOUT", vlane_split_axis, vlane_stride, mlir_dtype, dram_var, index_var, sram_var, sram_index_var,
                                dram_shape, final_tile_shape, attribute)
        self.reductions_suffix.writeline(DeferredLine(name, code))

    def set_tile_size(self, template_fusion_info, prologue=False):
        """Configure tile descriptor and related loop/reduction state based on template fusion info.

        왜 필요한가: 템플릿이 요구하는 타일 크기/벡터화 정보를 커널 상태에 반영하고, 리덕션일 경우 관련 루프 및 벡터 크기를 조정합니다.
        """
        tile_desc = template_fusion_info["dram_tile_desc"]
        if "dim_aliasing" in template_fusion_info:
            self.dim_aliasing = template_fusion_info["dim_aliasing"]

        if 'nr_rdim' in template_fusion_info and template_fusion_info['nr_rdim']==1:
            tile_desc.nr_rdim = 1
            numel_per_lane = tile_desc.get_numel_per_lane()
            r_tile_size = tile_desc.get_tile_size()[-1]
            nr_outer_loop = (numel_per_lane + r_tile_size-1) // r_tile_size
            tile_desc.vmap.forced_vec_size = nr_outer_loop * 32 # Why? Emprically selected, other option failed to functionality...

            self.reduction_fusion = True
            self.r_tile_size = tile_desc.get_tile_size()[-1]
            self.r_dim_size = template_fusion_info['r_dim_size']
            self.reduction_nr_outer_loop = nr_outer_loop
            self.reduction_loop_idx = "reduce_loop_idx"
            self.compute_body_loop.size = r_tile_size
            self.compute_body_loop.step = tile_desc.get_compute_vec_size() // nr_outer_loop
            self.reduction_body_loop = mlir_common.LoopLevel(self.reduction_loop_idx, nr_outer_loop)
        else:
            tile_desc.vmap.forced_vec_size = 64

            if prologue:
                self.prologue_compute_body_loop.size = tile_desc.get_numel_per_lane()
                self.prologue_compute_body_loop.step = tile_desc.get_compute_vec_size()
            else:
                self.compute_body_loop.size = tile_desc.get_numel_per_lane()
                self.compute_body_loop.step = tile_desc.get_compute_vec_size()
        return tile_desc

    def rename_indexing(self, index) -> sympy.Expr:
        """Apply dim_aliasing substitutions safely to avoid cyclic renames.

        필요성: dim aliasing을 적용할 때 이름 충돌(서로 바꾸는 케이스)을 피하기 위해 임시 이름을 사용하여 안전하게 치환합니다.
        """
        for dim_name, dim_aliased_name in self.dim_aliasing.items():
            index = index.subs(sympy.Symbol(dim_name), sympy.Symbol("tmp_"+dim_aliased_name))
        # To avoid this case ({"index0":"index1", "index1":"index0"})
        for dim_aliased_name in self.dim_aliasing.values():
            index = index.subs(sympy.Symbol("tmp_"+dim_aliased_name), sympy.Symbol(dim_aliased_name))
        return index

class MLIRTemplateCaller(CUDATemplateCaller):
    def __str__(self):
        return f"MLIRTemplateCaller(source_file={self.bmreq.source_file})"

    def call_name(self) -> str:
        return f"mlir_template_kernels.{self.name}"

class MLIRTemplate(KernelTemplate):
    index_counter = itertools.count()

    def __init__(self, name, input_nodes, layout, input_reorder = None):
        """
        Baseclass for MLIR Templates, derived from KernelTemplate. Not to be instantiated directly.

        Args:
            name (str): The name of the CUDATemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.

        """
        super().__init__(name)
        self.input_nodes = [node for node in input_nodes if node is not None]
        self.output_node: Buffer = Buffer("buf_out", layout)
        self.input_reorder = input_reorder
        self.layout = layout

    def generate(self, **kwargs) -> ChoiceCaller:
        kernel_name = f"mlir_{self.name}"
        with patch.object(V.graph, "get_dtype", self._fake_get_dtype(self.output_node)):
            kernel  = MLIRTemplateKernel(kernel_name=kernel_name, input_nodes=self.input_nodes, call_size=self.layout.size, kernel_group=None,
                                         outer_func_name=self.function_name if hasattr(self, 'function_name') else None,
                                         outer_func_render=self.outer_func_render if hasattr(self, 'outer_func_render') else None,
                                         kernel_arg_attributes=self.get_arg_attributes() if hasattr(self, 'get_arg_attributes') else None)
            code = self.render(kernel=kernel, **kwargs)

        kernel_hash_name = f"mlir_{self.name}_{next(self.index_counter)}"
        extra_args = []
        # create the BenchmarkRequest
        bmreq = MLIRBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: TemplateBuffer,
            prologue_nodes: Optional[List[IRNode]] = None,
            epilogue_nodes: Optional[List[IRNode]] = None,
            kernel_name: str = kernel_hash_name,
            kernel_group: Optional[mlir_common.MLIRWrapperKenrelGroup] = None
        ):
            kernel = MLIRTemplateKernel(
                kernel_name=kernel_name,
                input_nodes=self.input_nodes,
                call_size=self.layout.size,
                kernel_group=kernel_group,
                outer_func_name=self.function_name if hasattr(self, 'function_name') else None,
                outer_func_render=functools.partial(
                    self.outer_func_render,
                    kernel_name=kernel_name
                ) if hasattr(self, 'outer_func_render') else None,
                kernel_arg_attributes=self.get_arg_attributes() if hasattr(self, 'get_arg_attributes') else None
            )

            kwargs = {
                'kernel': kernel,
                'template_buffer_node': template_node,
                'epilogue_nodes': epilogue_nodes,
                'prologue_nodes': prologue_nodes,
            }
            render = functools.partial(
                kernel.render,
                template=self,
                kwargs=kwargs
            )
            tile_candidates = self.get_tile_candidates(**kwargs)[:extension_config.codegen_autotune_template_topk]
            return kernel, tile_candidates, render

        return MLIRTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
        )

    def get_tile_candidates(self, **kwargs):
        return []

    def render(self, **kwargs) -> str:
        raise NotImplementedError