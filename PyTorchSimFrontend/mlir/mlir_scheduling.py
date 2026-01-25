import os  
import math  
import sympy  
from functools import reduce  
import operator  
from sympy import symbols, sympify  
from PyTorchSimFrontend import extension_config  
from PyTorchSimFrontend.mlir.mlir_codegen_backend import MLIRKernel  

from torch._inductor import config  
from torch._inductor.scheduler import BaseScheduling, FusedSchedulerNode, SchedulerNode, BaseSchedulerNode 
from torch._inductor.utils import IndentedBuffer 
from torch._inductor.virtualized import V 
from torch._inductor.ir import LoopBody  
from torch._inductor import dependencies 

from . import mlir_common 
from . import mlir_lowering  


class MLIRScheduling(BaseScheduling): 
    count = 0 
    target_kernel = MLIRKernel  # 사용할 MLIR 커널을 지정.
    def __init__(self, scheduler):  
        self.scheduler = scheduler  # 스케줄러를 인스턴스 변수로 저장합니다.
        self.scheduler.can_fuse_origin = self.scheduler.can_fuse  # 원래의 fusion 가능성을 설정합니다.
        self.scheduler.can_fuse = self.can_fuse_with_exceptions  # 예외가 있는 fusion 가능성을 설정합니다.
        #self.scheduler.enter_context = self.enter_context_fixed  # FIXME. Inductor 버그 수정을 위한 몽키 패치
        self.kernel_group = mlir_common.MLIRWrapperKenrelGroup()  # MLIRWrapperKernelGroup 인스턴스를 생성합니다.
        self._ready_to_flush = False  # 플러시 준비 상태를 초기화합니다.
        self.outer_function = set()  # 외부 함수 집합을 초기화합니다.
        config.inplace_buffers = False  # FIXME. inout 커널 문제로 비활성화합니다.
        self.max_fusion_size = 5  # 최대 fusion 크기를 설정합니다.

    def can_fuse_with_exceptions(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:  # 예외가 있는 fusion 가능성을 확인하는 메서드
        base_template_node1 = [node for node in node1.get_nodes() if node.is_template()]  # node1의 템플릿 노드를 가져옵니다.
        base_template_node2 = [node for node in node2.get_nodes() if node.is_template()]  # node2의 템플릿 노드를 가져옵니다.
        if node1.get_device() != node2.get_device():  # 두 노드의 장치가 다르면
            return False  # fusion 불가능
        if not (isinstance(node1, (SchedulerNode, FusedSchedulerNode)) and isinstance(node2, (SchedulerNode, FusedSchedulerNode))):  # 두 노드가 스케줄러 노드가 아니면
            return False  # fusion 불가능

        if len(base_template_node1) == 1 and len(base_template_node2) == 0 and extension_config.CONFIG_FUSION_REDUCTION_EPILOGUE:  # 특정 조건을 만족하면
            from PyTorchSimFrontend.mlir.mlir_gemm_template import MLIRGemmTemplate  # GEMM 템플릿을 가져옵니다.
            from PyTorchSimFrontend.mlir.mlir_bmm_template import MLIRBMMTemplate  # BMM 템플릿을 가져옵니다.
            if (isinstance(base_template_node1[0].node.template, MLIRGemmTemplate) or isinstance(base_template_node1[0].node.template, MLIRBMMTemplate)) and node2.is_reduction():  # 매트릭스 곱셈과 축소의 경우
                # 매트릭스 곱셈/배치 매트릭스 곱셈 + 축소의 경우
                size_match = node1.get_nodes()[0].node.get_numel() == reduce(operator.mul, node2.get_nodes()[0].node.get_size(), 1) * reduce(operator.mul, node2.get_nodes()[0].node.get_reduction_size(), 1)  # 크기 일치 여부 확인
                target_symbol = symbols("r0")  # 기호 r0을 정의합니다.
                try:
                    stride = [i.strip()[:-1].split(",")[-1].strip() for i in str(node2.get_nodes()[0].node).split("\n") if "r0" in i][1]  # stride를 가져옵니다.
                    stride = int(sympify(stride).coeff(target_symbol))  # 기호를 사용하여 stride를 정수로 변환합니다.
                except:
                    return False  # 예외 발생 시 fusion 불가능

                # dim=-1로 fusion 불가능
                layout_possible = stride != 1  # 레이아웃 가능 여부 확인
                # 방향성 링크 확인
                dependency_check = node2.get_nodes()[0] in [node.node for node in base_template_node1[0].users]  # 의존성 확인
                dependency_size = all([i.get_numel() == node1.get_nodes()[0].node.get_numel() for i in node2.read_writes.reads])  # 의존성 크기 확인
                return size_match and layout_possible and dependency_check and dependency_size  # 모든 조건이 만족되면 fusion 가능
           
        # 프로로그 fusion의 경우
        if extension_config.CONFIG_FUSION_PROLOGUE and len(base_template_node1) == 0 and len(node1.get_nodes())==1 and len(base_template_node2) == 1:  # 특정 조건을 만족하면
            from PyTorchSimFrontend.mlir.mlir_gemm_template import MLIRGemmTemplate  # GEMM 템플릿을 가져옵니다.
            from PyTorchSimFrontend.mlir.mlir_bmm_template import MLIRBMMTemplate  # BMM 템플릿을 가져옵니다.
            target_node = base_template_node2[0].node  # 타겟 노드를 설정합니다.
            if target_node.origin_node is not None and hasattr(target_node.origin_node.target, "_name") and target_node.origin_node.target._name == 'aten::convolution':  # 특정 조건을 확인합니다.
                return False  # fusion 불가능
            if node1.is_reduction():  # node1이 축소 노드이면
                return False  # fusion 불가능
            if len(node1.read_writes.writes) != 1:  # node1의 쓰기 수가 1이 아니면
                return False  # fusion 불가능
            if node1.node not in target_node.inputs or any(["view" in str(ori) for ori in node1.node.origins]):  # node1이 타겟 노드의 입력이 아니면
                return False  # fusion 불가능

            # 현재 BMM과 MM만 프로로그 fusion을 지원합니다.
            if not isinstance(target_node.template, (MLIRBMMTemplate, MLIRGemmTemplate)):  # 타겟 노드가 BMM 또는 GEMM 템플릿이 아니면
                return False  # fusion 불가능
            # 이 엣지는 fusion하지 않습니다.
            if base_template_node2[0].group[1][0][0] == 1:  # 특정 조건을 확인합니다.
                return False  # fusion 불가능

            if list(node1.read_writes.writes)[0].name in [dep.name for dep in node2.read_writes.reads]:  # node1의 쓰기가 node2의 읽기와 일치하면
                node1 = self.revert_group(node1)  # node1의 그룹을 되돌립니다.
                return True  # fusion 가능

        return self.scheduler.can_fuse_origin(node1, node2)  # 기본 fusion 가능성 확인

    def _set_flush_status(self, status: bool):  # 플러시 상태를 설정하는 메서드 
        self._ready_to_flush = status  # 플러시 준비 상태를 업데이트합니다.
#flush = 스케줄러가 모아둔 커널 그룹을 실제 코드로 반환
    def can_fuse_vertical(self, node1, node2):  # 수직 fusion 가능성 확인
        return self.can_fuse_horizontal(node1, node2)  # 수평 fusion 가능성 확인

    def can_fuse_horizontal(self, node1, node2):  # 수평 fusion 가능성 확인
        if not extension_config.CONFIG_FUSION:  # fusion이 비활성화되어 있으면
            return False  # fusion 불가능
        if (len(node1.get_nodes())+ len(node2.get_nodes())) > self.max_fusion_size:  # 두 노드의 크기가 최대 fusion 크기를 초과하면
            return False  # fusion 불가능
        _, (vars1, reduce1) = node1.group  # node1의 변수와 축소 정보를 가져옵니다.
        _, (vars2, reduce2) = node2.group  # node2의 변수와 축소 정보를 가져옵니다.

        # 축소는 현재 지원되지 않습니다.
        if node1.is_reduction() and node2.is_reduction() and not node1.is_template() and not node2.is_template() and extension_config.CONFIG_FUSION_REDUCTION_REDUCTION:  # 두 노드가 축소 노드이면
            return vars1 == vars2 and reduce1 == reduce2 and node1.inverse_users == node2.inverse_users  # 변수와 축소 정보가 일치하면 fusion 가능
        if node1.is_reduction() or node2.is_reduction():  # 하나라도 축소 노드이면
            return False  # fusion 불가능

        # 두 템플릿 노드는 fusion할 수 없습니다.
        if node1.is_template() and node2.is_template():  # 두 노드가 템플릿이면
            return False  # fusion 불가능

        if '_unsafe_index' in node1.get_nodes()[0].node.origins or "_unsafe_index" in node2.get_nodes()[0].node.origins:  # 안전하지 않은 인덱스가 포함되어 있으면
            return False  # fusion 불가능

        # 템플릿 노드 fusion 확인
        if node1.is_template() or node2.is_template():  # 하나라도 템플릿이면
            # maxpool 템플릿 코드는 fusion하지 않습니다.
            from PyTorchSimFrontend.mlir.mlir_maxpool_template import MLIRMaxPoolTemplate  # MaxPool 템플릿을 가져옵니다.
            from PyTorchSimFrontend.mlir.mlir_bmm_template import MLIRBMMTemplate  # BMM 템플릿을 가져옵니다.
            from PyTorchSimFrontend.mlir.mlir_gemm_template import MLIRGemmTemplate  # GEMM 템플릿을 가져옵니다.
            template_node1 = next((n for n in node1.get_nodes() if n.is_template()), None)  # node1의 템플릿 노드를 가져옵니다.
            template_node2 = next((n for n in node2.get_nodes() if n.is_template()), None)  # node2의 템플릿 노드를 가져옵니다.

            if template_node1 and len(node1.get_nodes()) == 1 and isinstance(template_node1.node.template, MLIRMaxPoolTemplate) or \
               template_node2 and len(node2.get_nodes()) == 1 and isinstance(template_node2.node.template, MLIRMaxPoolTemplate):  # MaxPool 템플릿이면
                return False  # fusion 불가능

            # 포인트별 확인
            v1_total = math.prod(vars1) if len(vars1) else 0  # node1의 변수 곱을 계산합니다.
            v2_total = math.prod(vars2) if len(vars2) else 0  # node2의 변수 곱을 계산합니다.
            if v1_total != v2_total:  # 두 변수 곱이 일치하지 않으면
                return False  # fusion 불가능

            # 패턴 확인
            template_node, act_node = (template_node1, node2) if template_node1 else (template_node2, node1)  # 템플릿 노드와 활성 노드를 설정합니다.
            has_depedency = set(act_node.inverse_users) <= set(template_node.get_nodes())  # 의존성 확인
            if not has_depedency:  # 의존성이 없으면
                return False  # fusion 불가능

            # act_node.group 되돌리기: simplify_and_reorder()가 수정한 _body, _size, group
            if template_node.group != act_node.group:  # 그룹이 다르면
                # 이 경우 fusion하지 않습니다.
                if (isinstance(template_node.node.template, MLIRBMMTemplate) or isinstance(template_node.node.template, MLIRGemmTemplate)) and template_node.group[1][0][0] == 1:  # 특정 조건을 확인합니다.
                    return False  # fusion 불가능

                if list(template_node.group[1][0]) != list(act_node.get_nodes()[0].node.data.get_size()):  # 크기가 다르면
                    return False  # fusion 불가능
                self.revert_group(act_node)  # act_node의 그룹을 되돌립니다.
            return True  # fusion 가능

        # 요소별 fusion 확인
        if vars1 == vars2 and reduce1 == reduce2:  # 변수와 축소 정보가 일치하면
            return True  # fusion 가능
        return False  # fusion 불가능

    def revert_group(self, act_nodes, args=None, var_ranges=None):  # 그룹을 되돌리는 메서드
        for act_node in act_nodes.get_nodes():  # 각 활성 노드에 대해
            if args is None or var_ranges is None:  # 인자나 변수 범위가 주어지지 않으면
                args, var_ranges = dependencies.index_vars_no_squeeze(
                        act_node.node.data.get_size(), act_node.node.data.get_reduction_size(), prefix="q"
                )  # 변수 인덱스를 설정합니다.
            body = LoopBody(
                act_node.node.get_store_function(),
                (args if act_node.node.get_reduction_type() else args[:1]),
                var_ranges,
            )  # 루프 본체를 정의합니다.
            index_size = []  # 인덱스 크기 초기화
            reduce_size = []  # 축소 크기 초기화
            for v, s in var_ranges.items():  # 각 변수와 크기에 대해
                if v in args[0]:  # 인자에 변수가 포함되어 있으면
                    index_size.append(s)  # 인덱스 크기에 추가
                else:  # 그렇지 않으면
                    reduce_size.append(s)  # 축소 크기에 추가
            node_device = act_node.get_device()  # 노드의 장치를 가져옵니다.
            ranges = (index_size, reduce_size)  # 인덱스와 축소 크기 범위를 설정합니다.
            act_node._sizes, act_node._body, act_node.group = (ranges), body, (node_device, self.group_fn(ranges))  # 노드의 크기, 본체, 그룹을 업데이트합니다.

    def group_fn(self, sizes):  # 그룹 함수를 정의합니다.
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)  # 크기를 단순화하여 튜플로 반환합니다.

    def codegen_nodes(self, nodes):  # 노드에 대한 코드를 생성하는 메서드
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group  # 가장 큰 축소 그룹을 찾습니다.

        # 노드에 루프가 적어도 하나 있다고 가정합니다.
        # 그러나 인덕터가 그룹을 단순화하므로 루프가 없을 수도 있습니다.
        # 그런 경우, 더미 루프(크기=1)를 그룹에 추가합니다.
        if len(group) == 0:
            for idx, node in enumerate(nodes):
                if len(node.node.data.get_size()) == 0:
                    continue
                if len(reduction_group) != 0:
                    sym0, sym1 = sympy.Symbol("q0"), sympy.Symbol("q1")
                    args = [[sym0] + [sympy.Number(0)] * (len(node.node.data.get_size())-1), [sym1]]
                    var_ranges = {sym0: sympy.Number(1), sym1: reduction_group[0]}
                else:
                    sym0 = sympy.Symbol("q0")
                    args = [[sym0] + [sympy.Number(0)] * (len(node.node.data.get_size())-1), []]
                    var_ranges = {sym0: sympy.Number(1)}
                self.revert_group(node, args, var_ranges)  # 노드 그룹을 되돌립니다.
            _, (group, reduction_group) = max(
                nodes, key=lambda x: int(x.is_reduction())
            ).group  # 다시 가장 큰 축소 그룹을 찾습니다.

        ex_kernel = self.target_kernel(kernel_group=self.kernel_group)  # 타겟 커널 인스턴스를 생성합니다.
        ex_kernel.kernel_group = self.kernel_group  # 커널 그룹을 설정합니다.

        kernel_name_candidate = f"extension_kernel_{MLIRScheduling.count}"  # 커널 이름 후보를 설정합니다.
        MLIRScheduling.count += 1  # 클래스 변수를 증가시킵니다.
        src_code = ex_kernel.codegen_nodes(nodes, kernel_name_candidate)  # 노드에 대한 코드를 생성합니다.
        kernel_name = self.define_kernel(src_code, kernel_name_candidate, ex_kernel.vector_lane,
                           ex_kernel.spad_info, origins= {str(i) for i in nodes[0].node.origins})  # 커널을 정의합니다.
        ex_kernel.call_kernel(kernel_name)  # 커널을 호출합니다.
        _, args, _, _ = ex_kernel.args.mlir_argdefs()  # 커널 인자 정보를 가져옵니다.
        args = ", ".join(args)  # 인자를 문자열로 변환합니다.
        eager_mode = int(os.environ.get('TOGSIM_EAGER_MODE', default=False))  # EAGER 모드 설정을 가져옵니다.
        if (eager_mode):
            V.graph.wrapper_code.writeline(
                f"yield ({kernel_name}, ({args}))"
            )  # EAGER 모드에서 커널 호출을 기록합니다.
        self._set_flush_status(True)  # 플러시 준비 상태를 True로 설정합니다.


    def ready_to_flush(self):  # 플러시 준비 상태 확인
        return self._ready_to_flush  # 현재 플러시 준비 상태를 반환합니다.

    def codegen_sync(self):  # 동기화 코드 생성을 위한 메서드
        pass  # 현재 구현은 비워둡니다.

    def flush(self):  # 플러시 메서드
        self.kernel_group.codegen_define_and_call(V.graph.wrapper_code)  # 커널 그룹에 대한 정의 및 호출을 생성합니다.
        self.kernel_group = mlir_common.MLIRWrapperKenrelGroup()  # 새로운 MLIRWrapperKernelGroup 인스턴스를 생성합니다.
        self._set_flush_status(False)  # 플러시 준비 상태를 False로 설정합니다.

    def define_function(self, kernel):  # 커널에 대한 함수를 정의하는 메서드
        partial_code, function_name = kernel.def_function()  # 커널에서 부분 코드를 가져옵니다.
        if partial_code is not None and function_name not in self.outer_function:  # 유효한 코드와 함수 이름 확인
            with V.set_kernel_handler(kernel):  # 커널 핸들러 설정
                code = partial_code.finalize()  # 부분 코드를 최종화합니다.
                wrapper = V.graph.wrapper_code  # 그래프의 래퍼 코드를 가져옵니다.
                wrapper.header.writeline(code)  # 헤더에 코드를 기록합니다.
                self.outer_function.add(function_name)  # 외부 함수 집합에 함수 이름을 추가합니다.

    def define_kernel(self, src_code, kernel_name, vector_lane, spad_info, loop_size=None, origins={}):  # 커널 정의 메서드
        wrapper = V.graph.wrapper_code  # 그래프의 래퍼 코드를 가져옵니다.
        if src_code in wrapper.src_to_kernel:  # 소스 코드가 이미 등록되어 있으면
            kernel_name = wrapper.src_to_kernel[src_code]  # 기존 커널 이름을 사용합니다.
        else:
            wrapper.src_to_kernel[src_code] = kernel_name  # 새로운 소스 코드를 등록합니다.

            codecache_def = IndentedBuffer()  # 코드 캐시 정의를 위한 버퍼 생성
            codecache_def.writeline(f"custom_async_compile.mlir('''{src_code}''', ")  # MLIR 커널 컴파일 호출
            codecache_def.writeline(f"vectorlane_size={vector_lane},")  # 벡터 레인 크기 설정
            codecache_def.writeline(f"loop_size={loop_size},")  # 루프 크기 설정
            codecache_def.writeline(f"spad_info={spad_info},")  # SPAD 정보 설정
            codecache_def.writeline(f"origins={origins},")  # 기원 정보 설정
            codecache_def.writeline("arg_attributes=arg_attributes,")  # 인자 속성 설정
            codecache_def.writeline(f"vlen={extension_config.vpu_vector_length_bits})")  # VPU 벡터 길이 설정
            wrapper.define_kernel(kernel_name, codecache_def.getvalue(), cuda=False)  # 커널을 정의합니다.
        return kernel_name  # 커널 이름 반환

    def codegen_template(self, template_node, epilogue_nodes):  # 템플릿 코드 생성을 위한 메서드
        # 프로로그 패턴 처리
        prologue_nodes = []  # 프로로그 노드 초기화
        if not template_node.is_template():  # 템플릿 노드가 아니면
            epilogue_nodes = [template_node] + epilogue_nodes  # 에필로그 노드에 템플릿 노드를 추가합니다.
            for i, node in enumerate(epilogue_nodes):  # 각 에필로그 노드에 대해
                if node.is_template():  # 노드가 템플릿이면
                    template_node = node  # 템플릿 노드를 업데이트합니다.
                    prologue_nodes = epilogue_nodes[:i]  # 프로로그 노드를 설정합니다.
                    epilogue_nodes = epilogue_nodes[i+1:]  # 나머지 에필로그 노드를 업데이트합니다.
                    break

        # 템플릿 코드 생성
        template_buffer = template_node.node  # 템플릿 노드의 버퍼를 가져옵니다.
        kernel, tile_candidates, render = template_buffer.make_kernel_render(template_buffer, prologue_nodes=prologue_nodes, epilogue_nodes=epilogue_nodes, kernel_group=self.kernel_group)  # 커널 렌더링을 위한 설정
        _, _, _, kernel.buffer_types = self.kernel_group.args.mlir_argdefs()  # 버퍼 타입 정보를 가져옵니다.
        src_code = kernel.codegen_nodes(tile_candidates, render, template_node, prologue_nodes, epilogue_nodes)  # 템플릿 노드에 대한 코드를 생성합니다.

        with V.set_kernel_handler(kernel):  # 커널 핸들러 설정
            kernel_name = self.define_kernel(src_code, kernel.kernel_name, kernel.vector_lane, kernel.spad_info,
                                             kernel.loop_size, origins={str(i) for i in template_node.node.origins})  # 커널 정의
            self.define_function(kernel)  # 커널에 대한 함수 정의

        kernel.call_kernel(kernel_name)  # 커널 호출
        V.graph.removed_buffers |= kernel.removed_buffers  # 제거된 버퍼 업데이트
        _, args, _, _ = self.kernel_group.args.mlir_argdefs()  # 커널 인자 정보 가져오기
        eager_mode = int(os.environ.get('TOGSIM_EAGER_MODE', default=False))  # EAGER 모드 설정 가져오기
        if (eager_mode):
            target_kernel_name = kernel_name if kernel.outer_func_name is None else kernel.outer_func_name + f"_{len(args)}"  # 타겟 커널 이름 설정
            args = ", ".join(args)  # 인자를 문자열로 변환
            V.graph.wrapper_code.writeline(
                f"yield ({target_kernel_name}, ({args}))"
            )  # EAGER 모드에서 커널 호출 기록
        self._set_flush_status(True)  # 플러시 준비 상태를 True로 설정

    def enter_context_fixed(self, node):  # 컨텍스트 진입을 위한 고정 메서드
        def get_order(n):  # 노드 순서를 가져오는 내부 함수
            if n not in self.scheduler.origin_to_index:  # 노드가 인덱스에 없으면
                self.scheduler.origin_to_index.update({n: i for i, n in enumerate(n.graph.nodes)})  # 노드 인덱스를 업데이트합니다.
            return self.scheduler.origin_to_index[n]  # 노드의 인덱스를 반환합니다.

        origins = [(get_order(e), idx, e) for n in node.get_nodes() for idx, e in enumerate(n.node.origins)]  # 노드 기원 정보 가져오기
        if origins:  # 기원 정보가 있으면
            _, _, last = max(origins)  # 가장 마지막 기원 노드를 찾습니다.
            V.graph.wrapper_code.enter_context(last)  # 컨텍스트 진입
