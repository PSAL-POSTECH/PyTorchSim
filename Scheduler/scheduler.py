from typing import List
import os
import sys
import numpy as np
import torch
from pathlib import Path
import importlib.util
from PyTorchSimFrontend.extension_codecache import hash_prefix
from Simulator.simulator import TOGSimulator
from PyTorchSimFrontend import extension_config

# Configure logger for Scheduler module
logger = extension_config.setup_logger()


def import_module_from_path(module_name, path):
    module_path = Path(path)  # Convert to Path object for safety
    if not module_path.exists() or not module_path.is_file():
        raise FileNotFoundError(f"No such file: '{module_path}'")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Could not load module from path: '{module_path}'")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

def poisson_request_generator(lambda_requests, max_msec_time=None):
    current_time = 0.0 # msec

    yield 0
    while max_msec_time is None or current_time < max_msec_time:
        inter_arrival_time = np.random.exponential(scale=1000 / lambda_requests)
        current_time += inter_arrival_time

        if max_msec_time is not None and current_time > max_msec_time:
            break

        yield current_time

class Request:
    """ Each request has model name, it's own id, and requested time. """
    request_id = 0
    QUEUED     = 1
    RUNNING    = 2
    INCREMENT  = 3
    FINISHED   = 4
    def __init__(self, model:str, batchable_input_tensor : List[torch.Tensor],
                 shared_input_tensor: List[torch.tensor], request_queue_idx=0) -> None:
        self.model = model
        self.batchable_input_tensor = batchable_input_tensor
        self.shared_input_tensor = shared_input_tensor
        self.arrival_time = None
        self.start_time = []
        self.finish_time = []
        self.state = self.QUEUED
        self.id = self.allocate_id()
        self.request_queue_idx = request_queue_idx

    def allocate_id(self):
        allocated_id = Request.request_id
        Request.request_id += 1
        return allocated_id

    def set_start(self, start_time):
        self.state = self.RUNNING
        self.start_time.append(start_time)

    def set_finished(self, finish_time):
        self.state = self.FINISHED
        self.finish_time.append(finish_time)

    def get_latency(self):
        # Todo. Provide Toke-By-Token
        if self.state == self.FINISHED:
            turnaround_time = self.finish_time[-1] - self.arrival_time
        else:
            turnaround_time = None

        if self.start_time:
            response_time = self.start_time[0] - self.arrival_time
        else:
            response_time = None

        if self.start_time and self.finish_time:
            tbt_time = [i-j for i,j in zip(self.finish_time, self.start_time)]
        else:
            tbt_time = []

        return turnaround_time, response_time, tbt_time

    def free_memory(self):
        """ Free memory resources that are allocated for handle this request """
        return

    def __str__(self) -> str:
        return f"Request{self.id} Model: '{self.model}', Arrival: {self.arrival_time}, Start: {self.start_time}, End: {self.finish_time}, State: {self.state}, Partion: {self.request_queue_idx}"

class RequestReturn:
    INCREMENT = 0
    FINISHED = 1
    def __init__(self, state) -> None:
        self.state = state

    def is_finished(self):
        return self.state == self.FINISHED

    def is_increment(self):
        return self.state == self.INCREMENT

class SchedulerDNNModel:
    MODEL_MAP = {}
    def __init__(self, batched_req : List[Request], partition_idx) -> None:
        self.model_name = batched_req[0].model
        self.batched_req = batched_req
        self.args = None
        self.model = self.find_model(self.model_name)
        self.partition_idx = partition_idx

    def find_model(self, model_name : str):
        if model_name in SchedulerDNNModel.MODEL_MAP:
            return SchedulerDNNModel.MODEL_MAP[model_name]
        else:
            raise KeyError(f'[Scheduler] Requested model "{model_name}" is not registered...')

    def get_batchable_input(self):
        batched_input_tensor = []
        for i in range(len(self.batched_req[0].batchable_input_tensor)):
            tensor_list = [req.batchable_input_tensor[i] for req in self.batched_req]
            batched_input_tensor.append(torch.concat(tensor_list, dim=0))
        return batched_input_tensor

    def get_shared_input(self):
        return self.batched_req[0].shared_input_tensor

    def get_input(self):
        return self.get_batchable_input() + self.get_shared_input()

    def __str__(self):
        return f"DNN Model: {self.model_name}, Partion idx: {self.partition_idx} Req: {self.batched_req[0]}"

    @staticmethod
    def register_model(model_name : str, compiled_model):
        SchedulerDNNModel.MODEL_MAP[model_name] = compiled_model

class PyTorchSimRunner:
    PARTITION_BUSY = 0
    PARTITION_IDLE = 1
    SELECT_NOTHING = 2
    NPU_MODULE = None
    def __init__(self, tog_simulator : TOGSimulator, num_partion=1) -> None:
        self.module = self.setup_device()
        self.num_partion = num_partion
        self.launch_model_dicts = []
        self.nested_launch_model_dicts = []
        self.partition_state = []
        for i in range(self.num_partion):
            self.launch_model_dicts.append({})
            self.nested_launch_model_dicts.append({})
            self.partition_state.append(self.PARTITION_IDLE)

        self.finish_req_dict = {}
        self.tog_simulator = tog_simulator

        # Dry run for compile and create generator
        os.environ["TOGSIM_EAGER_MODE"] = "1"

    @classmethod
    def setup_device(cls):
        if cls.NPU_MODULE is not None:
            return cls.NPU_MODULE

        try:
            from torch._inductor.codegen.common import register_backend_for_device
            from PyTorchSimFrontend.mlir.mlir_codegen_backend import ExtensionWrapperCodegen
            from PyTorchSimFrontend.mlir.mlir_scheduling import MLIRScheduling
        except ImportError as e:
            logger.error(f"Failed to import torch_openreg: {e}")
            logger.error("Please ensure PyTorchSimDevice2 is installed: pip install -e PyTorchSimDevice2")
            raise

        register_backend_for_device(
            "npu",
            lambda scheduling: MLIRScheduling(scheduling),
            ExtensionWrapperCodegen
        )

        cls.NPU_MODULE = torch.npu
        return cls.NPU_MODULE

    def submit(self, batched_req, partition_idx) -> List[RequestReturn]:
        # FIXME. Construct SchedulerDNNModel
        batched_req_model = self.get_compiled_model(batched_req, partition_idx)
        self.prepare_model(batched_req_model)

    def get_compiled_model(self, batched_req: List[Request], request_queue_idx):
        compiled_model = SchedulerDNNModel(batched_req, request_queue_idx)
        return compiled_model

    def is_partition_idle(self, partition_idx):
        return len(self.launch_model_dicts[partition_idx]) == 0

    def is_any_idle(self, skip_list):
        return any([self.is_partition_idle(i) and not skip_list[i] for i in range(self.num_partion)])

    def is_all_idle(self):
        return all([self.is_partition_idle(i) for i in range(self.num_partion)])

    def prepare_model(self, req_model: SchedulerDNNModel):
        result_path = os.path.join(extension_config.CONFIG_TORCHSIM_LOG_PATH, "togsim_result", req_model.model_name)
        os.makedirs(result_path, exist_ok=True)
        index = str(len(os.listdir(result_path)))

        # Prepare input tensor
        input_tensor_list = req_model.get_input()
        input_tensor_list = [input_tensor.to(device=self.module.custom_device()) for input_tensor in input_tensor_list]

        # This model-call will return generator
        ret = req_model.model(*input_tensor_list)
        self.launch_model_dicts[req_model.partition_idx][req_model] = ret

    def finish_model(self, model : SchedulerDNNModel, output : torch.Tensor):
        for req in model.batched_req:
            # TODO. finish time
            self.finish_req_dict[req] = RequestReturn(RequestReturn.FINISHED)

    def prepare_launch_kernel(self, kernel, inputs):
        result_path, runtime_path, _ = kernel(*inputs)
        onnx_path = os.path.join(result_path, "tile_graph.onnx")

        attribute_path = os.path.join(runtime_path, "attribute")
        attribute_path = self.tog_simulator.create_attribute_file(attribute_path, inputs)
        return onnx_path, attribute_path

    def launch_kernel(self, current_cycle, partion_idx=0):
        # Check partition is busy
        if self.partition_state[partion_idx] != self.PARTITION_IDLE:
            return self.partition_state[partion_idx]
        result = self.select_kernel(partion_idx)
        if result == self.SELECT_NOTHING:
            return self.SELECT_NOTHING
        kernel, inputs = result
        if not isinstance(kernel, str):
            onnx_path, attribute_path = self.prepare_launch_kernel(kernel, inputs)
        else:
            onnx_path, attribute_path = kernel, inputs
        self.partition_state[partion_idx] = self.PARTITION_BUSY
        return self.tog_simulator.launch(onnx_path, attribute_path, current_cycle, partion_idx)

class FIFORunner(PyTorchSimRunner):
    def __init__(self, tog_simulator: TOGSimulator, num_partion=1) -> None:
        super().__init__(tog_simulator, num_partion)

    def select_kernel(self, partition_idx):
        while len(self.nested_launch_model_dicts[partition_idx]) or len(self.launch_model_dicts[partition_idx]):
            if len(self.nested_launch_model_dicts[partition_idx]):
                target_dict = self.nested_launch_model_dicts
            else:
                target_dict = self.launch_model_dicts

            # Select FIFO manner
            req, target_model = next(iter(target_dict[partition_idx].items()))
            try:
                kernel, inputs = next(target_model)

                # For extern call
                if isinstance(kernel, str):
                    return kernel, inputs

                # For convolution...
                if not hasattr(kernel, "future"):
                    nested_gen = kernel(*inputs)
                    self.nested_launch_model_dicts[partition_idx] = {req : nested_gen}
                    kernel, inputs = \
                        next(self.nested_launch_model_dicts[partition_idx][req])
                return kernel, inputs
            except StopIteration as e:
                # Retry
                if target_dict == self.launch_model_dicts:
                    self.finish_model(req, e.value)
                del target_dict[partition_idx][req]
        # No proper kernel now
        return self.SELECT_NOTHING

class RoundRobinRunner(PyTorchSimRunner):
    def __init__(self, tog_simulator: TOGSimulator, num_partion=1) -> None:
        super().__init__(tog_simulator, num_partion)
        self.next_pointer = None

    def select_kernel(self, partition_idx):
        while len(self.nested_launch_model_dicts[partition_idx]) or len(self.launch_model_dicts[partition_idx]):
            if len(self.nested_launch_model_dicts[partition_idx]):
                target_dict = self.nested_launch_model_dicts
            else:
                target_dict = self.launch_model_dicts

            req_list = list(target_dict[partition_idx].keys())
            # Select RR manner
            if self.next_pointer is None or self.next_pointer not in req_list:
                req = req_list[0]
                pos = 0
            else:
                req = self.next_pointer
                pos = req_list.index(req)

            # Set Next pointer
            if pos + 1 < len(req_list):
                self.next_pointer = req_list[pos+1]
            else:
                self.next_pointer = req_list[0]

            target_model = self.launch_model_dicts[partition_idx][req]
            try:
                kernel, inputs = next(target_model)

                # For convolution...
                if not hasattr(kernel, "future"):
                    nested_gen = kernel(*inputs)
                    self.nested_launch_model_dicts[partition_idx] = {req : nested_gen}
                    kernel, inputs = \
                        next(self.nested_launch_model_dicts[partition_idx][req])
                return kernel, inputs
            except StopIteration as e:
                # Retry
                if target_dict == self.launch_model_dicts:
                    self.finish_model(req, e.value)
                del self.launch_model_dicts[partition_idx][req]
        # No proper kernel now
        return self.SELECT_NOTHING

class Scheduler:

    FIFO_ENGINE = 0
    RR_ENGINE = 1
    def __init__(self, num_request_queue=1, max_batch=1, engine_select=FIFO_ENGINE, togsim_config=extension_config.CONFIG_TOGSIM_CONFIG) -> None:
        self.current_cycle = 0
        self.max_batch = max_batch
        self.num_request_queue = num_request_queue
        self.request_queue : List[List[Request]] = []
        for i in range(self.num_request_queue):
            self.request_queue.append([])
        self.finish_queue : List[Request] = []

        togsim_path = os.path.join(extension_config.CONFIG_TORCHSIM_DIR, "TOGSim")
        self.tog_simulator = TOGSimulator(togsim_path, togsim_config)
        if self.tog_simulator.config_yaml['pytorchsim_timing_mode'] == 0:
            # Scheduler requires timing mode to be enabled (pytorchsim_timing_mode != 0).
            logger.error(f"pytorchsim_timing_mode is set to 0 in config file '{togsim_config}'. ")
            logger.error(f"Scheduler requires timing mode to be enabled (pytorchsim_timing_mode != 0).")
            exit(0)

        os.environ['TOGSIM_CONFIG'] = togsim_config
        self.tog_simulator.interactive_simulation()
        if engine_select == Scheduler.FIFO_ENGINE:
            self.execution_engine = FIFORunner(self.tog_simulator, self.num_request_queue)
        elif engine_select == Scheduler.RR_ENGINE:
            self.execution_engine = RoundRobinRunner(self.tog_simulator, self.num_request_queue)
        else:
            logger.error(f"Not supported engine type {engine_select}")
            exit(1)

    def add_request(self, request: Request, request_time=-1):
        """register model at timestamp time
            request_time : msec
        """
        request_time = self.current_time() if request_time == -1 else request_time
        request.arrival_time = request_time
        self.request_queue[request.request_queue_idx].append(request)

    def request_empty(self, request_queue_idx):
        return len(self.request_queue[request_queue_idx])==0

    def select(self, request_queue_idx=0) -> List[Request]:
        """
        Select 1 request from request_queue in FCFS manner.
        If there is no proper request, return None
        """
        candidate_req = []
        if not self.request_queue[request_queue_idx]:
            return candidate_req
        for req in self.request_queue[request_queue_idx]:

            if self.msec_to_cycle(req.arrival_time) <= self.current_cycle and req.state == Request.QUEUED:
                candidate_req.append(req)

                # Stop batching
                if self.max_batch <= len(candidate_req):
                    break
        return candidate_req

    def next_request_time(self, request_queue_idx=0):
        for req in self.request_queue[request_queue_idx]:
            if req.state == Request.QUEUED:
                return req, req.arrival_time
        return None, -1

    def nearest_next_reqeust_time(self):
        nearest_req = None
        nearest_arrival_time = -1
        for i in range(self.num_request_queue):
            req, arrival_time = self.next_request_time(i)
            if nearest_arrival_time == -1 and arrival_time != -1:
                nearest_req = req
                nearest_arrival_time = arrival_time
            elif arrival_time != -1 and nearest_arrival_time > arrival_time:
                nearest_req = req
                nearest_arrival_time = arrival_time
        return nearest_req, nearest_arrival_time

    def finish_request(self, req : Request):
        req.set_finished(self.current_time())

        # Free resources
        req.free_memory()

        # Move to finish queue
        self.finish_queue.append(req)
        self.request_queue[req.request_queue_idx].remove(req)
        turnaround_time, response_time, tbt_time = req.get_latency()
        logger.info(
            f"[Request-{req.id} finished] partition: {req.request_queue_idx} arrival_time: "
            f"{req.arrival_time} start_time: {req.start_time[0]} turnaround latency: {turnaround_time}, "
            f"response time: {response_time} tbt_time: {tbt_time}"
        )

    def per_schedule(self, request_queue_idx):
        # Wait partition is idle
        if not self.execution_engine.is_partition_idle(request_queue_idx):
            return False

        request_list = self.select(request_queue_idx)
        if not request_list:
            return False

        logger.info(f"[Request issue] partition: {request_queue_idx} batch size: {len(request_list)}")
        for req in request_list:
            req.set_start(self.current_time())
            logger.info(
                f"[Request-{req.id} issue] partition: {req.request_queue_idx} "
                f"arrival_time: {req.arrival_time} start_time: {req.start_time[0]}"
            )
        # Submit batched request
        self.execution_engine.submit(request_list, request_queue_idx)

        return True

    def check_finish_request(self):
        # Check finished request
        while self.execution_engine.finish_req_dict:
            req, req_ret = next(iter(self.execution_engine.finish_req_dict.items()))
            self.finish_request(req)
            del self.execution_engine.finish_req_dict[req]

    def schedule(self):
        # Try schedule all request queue
        result = []
        for i in range(self.num_request_queue):
            result.append(self.per_schedule(i))

        # Try move to next nearest request time
        next_req, next_time = self.nearest_next_reqeust_time()
        if next_req is None and self.execution_engine.is_all_idle():
            # No request remained...
            return

        # Need to forward the time until next_arrival_time
        if self.execution_engine.is_all_idle():
            reason = self.tog_simulator.until(self.msec_to_cycle(next_time))
            self.current_cycle = self.tog_simulator.cycle()
        else:
            self.run(next_time)
        return

    def run(self, until_time):
        req_empty_info = [self.request_empty(i) for i in range(self.execution_engine.num_partion)]
        def execute_cycle():
            launch_ret_info = []
            for i in range(self.execution_engine.num_partion):
                if self.execution_engine.partition_state[i] == PyTorchSimRunner.PARTITION_IDLE:
                    ret = self.execution_engine.launch_kernel(self.current_cycle, i)
                    launch_ret_info.append(ret)

            self.check_finish_request()
            # Check if the stop condition is met
            if self.execution_engine.is_any_idle(req_empty_info) or self.execution_engine.is_all_idle(): # Ignore empty request queue
                return []

            # Schedule jobs and update the current time
            result_list = self.tog_simulator.until(self.msec_to_cycle(until_time))
            self.current_cycle = self.tog_simulator.cycle()

            for core_idx in result_list:
                # Kernel is finished. So set idle state
                self.execution_engine.partition_state[core_idx] = PyTorchSimRunner.PARTITION_IDLE

            return result_list

        if self.current_cycle >= self.msec_to_cycle(until_time):
            until_time = -1

        if until_time == -1:
            while not self.execution_engine.is_any_idle(req_empty_info):
                result = execute_cycle()
                req_empty_info = [self.request_empty(i) for i in range(self.execution_engine.num_partion)]
                # if result is not -1, schedule new request
                if len(result)==0:
                    break

        else:
            while self.current_cycle <= self.msec_to_cycle(until_time) and not self.execution_engine.is_all_idle():
                result = execute_cycle()
                # if result is not -1, schedule new request
                if len(result)==0:
                    break
        return

    def is_request_queue_empty(self):
        result = True
        for i in range(self.num_request_queue):
            result = result and (not len(self.request_queue[i]))
        return result

    def is_finished(self):
        if self.is_request_queue_empty() and self.execution_engine.is_all_idle():
            self.tog_simulator.wait()
            return True
        return False

    def current_time(self):
        return self.cycle_to_msec(self.current_cycle)

    def cycle_to_msec(self, cycle):
        freq = self.tog_simulator.get_core_freq()
        return cycle / (freq  / 1000)

    def msec_to_cycle(self, msec):
        # We treat -1 as special time
        if (msec == -1):
            return msec

        freq = self.tog_simulator.get_core_freq()
        return int(msec * (freq / 1000))
