import os
import subprocess
import math
import struct
from datetime import datetime
import random
import torch
import numpy as np
import hashlib
from torch._inductor.select_algorithm import ExternKernelChoice
from torch._inductor.codecache import get_hash
from AsmParser.tog_generator import tog_generator
from torch._inductor.codecache import write
from PyTorchSimFrontend.extension_codecache import get_write_path
from PyTorchSimFrontend import extension_config
from Simulator.simulator import TOGSimulator, TORCH_TO_NUMPY

graph_template = {
    0: {
        "node_id": 0,
        "node_name": "root",
        "node_type": 0,
        "parents": [],
        "children": [1]
    },
    1: {
        "node_id": 1,
        "node_name": "loopNode",
        "node_type": 2,
        "parents": [0],
        "children": [2],
        "loop_index": "loop_arg000",
        "loop_start": 0,
        "loop_end": 4,  # FIXME. this is a trick that generate multiple tile.
        "loop_step": 1,
        "loop_type": "outer_loop"
    },
    2: {
        "node_id": 2,
        "node_name": "stonneNode",
        "node_type": 5,
        "parents": [1],
        "children": [],
    }
}

class MLIRExternKernelChoice(ExternKernelChoice):
    def call_name(self):
        return f"torch.ops.extension_op.{self.name}"

custom_lib = torch.library.Library("extension_op", "DEF")

def calculate_sparsity(tensor):
    total_elements = tensor.numel()
    zero_elements = torch.sum(tensor.cpu() == 0)
    sparsity_ratio = zero_elements / total_elements * 100
    return math.ceil(sparsity_ratio.item())

def generate_outer_product_matrix(a, b, M, K, N, prefix, dir_path):
    # Generating matrix A
    data_width = 4
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    value_pointer = os.path.join(dir_path, f'{prefix}_outerproduct_gemm_mem.ini')
    rowA_pointer = os.path.join(dir_path, f'{prefix}_outerproduct_gemm_rowpointerA.in')
    colA_pointer = os.path.join(dir_path, f'{prefix}_outerproduct_gemm_colpointerA.in')
    rowB_pointer = os.path.join(dir_path, f'{prefix}_outerproduct_gemm_rowpointerB.in')
    colB_pointer = os.path.join(dir_path, f'{prefix}_outerproduct_gemm_colpointerB.in')

    with open(value_pointer, "w") as fd, open(rowA_pointer, "w") as rpA, open(colA_pointer, "w") as cpA, open(rowB_pointer, "w") as rpB, open(colB_pointer, "w") as cpB:
        #generating matrixA
        n_nonzeros=0
        for k in range(K):  # col major
            initial_values=0
            rpA.write(str(n_nonzeros)+","); # writing the index of A
            for m in range(M):
                if(a_cpu[m, k]):  # value is nonzero
                    if((m==(M-1)) and (k==(K-1))):
                        cpA.write(str(m))
                    else:
                        cpA.write(str(m)+","); #writing the row index
                    initial_values+=1
                    value = a_cpu[m, k]
                    ba = bytearray(struct.pack(">f", value))  # generating list of bytes
                    my_int = int.from_bytes(ba, "big")
                    fd.write(str(my_int))
                    fd.write(",")
                    n_nonzeros+=1
        rpA.write(str(n_nonzeros))
        address_matrix_b=n_nonzeros*data_width
        #Generating matrix B
        n_nonzeros=0
        for k in range(0,K):  # Row major
            initial_values=0
            rpB.write(str(n_nonzeros)+","); # writing the index of A
            for n in range(0,N):
                if(b_cpu[k, n]):  # value is nonzero
                    if((k==(K-1)) and (n==(N-1))):
                        cpB.write(str(n))
                    else:
                        cpB.write(str(n)+","); #writing the row index

                    initial_values+=1
                    value = b_cpu[k, n]
                    ba = bytearray(struct.pack(">f", value))  # generating list of bytes
                    my_int = int.from_bytes(ba, "big")
                    fd.write(str(my_int))
                    fd.write(",")
                    n_nonzeros+=1

        rpB.write(str(n_nonzeros))
        fd.write(str(0)) # Adding a final 0 to the memory which will never be used. This is just to avoid having a last comma.
        address_matrix_c=address_matrix_b+(n_nonzeros*data_width)
    return 0, address_matrix_b, address_matrix_c

def generate_inner_product_matrix(a, b, M, K, N, file_name, in_file_bitmap_a, in_file_bitmap_b):
    data_width = 4
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    matrixA_size=int(M*K)
    matrixB_size=int(N*K)
    matrixC_size=int(M*N)

    random.seed(a=0, version=2)

    address_matrix_a = 0
    with open(file_name, "w") as fd, open(in_file_bitmap_a, "w") as fbA, open(in_file_bitmap_b, "w") as fbB:
        #generating matrixA
        n_nonzeros=0
        for m in range(M):  # Row major
            for k in range(K):
                is_sparse = a_cpu[m,k]
                if(torch.isclose(is_sparse, torch.zeros(1), atol=1e-1)):
                    if((m==(M-1)) and (k==(K-1))):
                        fbA.write(str(1))
                    else:
                        fbA.write(str(1)+","); #writing a 1 in bitmap
                    ba = bytearray(struct.pack(">f", is_sparse))  # generating list of bytes
                    my_int = int.from_bytes(ba, "big")
                    fd.write(str(my_int))
                    fd.write(",")
                    n_nonzeros+=1
                else:
                    if((m==(M-1)) and (k==(K-1))): # this is to insert a comma
                        fbA.write(str(0))
                        # note no data element is inserted in this case
                    else:
                        # note no data element is inserted in this case
                        fbA.write(str(0)+",")

        address_matrix_b=n_nonzeros*data_width
        #Generating matrix B
        n_nonzeros=0
        bitmapB=list(range(0,matrixB_size))
        for n in range(0,N):  # Row major
            for k in range(0,K):
                is_sparse = b_cpu[k,n]
                if(torch.isclose(is_sparse, torch.zeros(1), atol=1e-1)):  # value is generated
                    bitmapB[k*N+n]=1
                    ba = bytearray(struct.pack(">f", float(is_sparse)))  # generating list of bytes
                    my_int = int.from_bytes(ba, "big")
                    fd.write(str(my_int))
                    fd.write(",")
                    n_nonzeros+=1
                else:
                    # no data element is inserted in this case
                    bitmapB[k*N+n]=0; #writing a 0
        # writing the bitmapB in the appropiate order
        for i in range(0, matrixB_size):
            fbB.write(str(bitmapB[i]))
            if(i < (matrixB_size-1)):
                fbB.write(",")
        
        fd.write(str(0)) # Adding a final 0 to the memory which will never be used. This is just to avoid having a last comma.
        address_matrix_c=address_matrix_b+(n_nonzeros*data_width)
    print("Offset matrix A: "+str(address_matrix_a))
    print("Offset matrix B: "+str(address_matrix_b))
    print("Offset matrix C: "+str(address_matrix_c))
    return address_matrix_a, matrixA_size, matrixA_size+matrixB_size

def prepare_outer_product_matrix(a, b, out):
    M, K, N = a.shape[0], b.shape[0], b.shape[1]

    prefix = datetime.now().strftime("%m%d%H%M%S%f")
    w_sparsity = calculate_sparsity(a)
    x_sparsity = calculate_sparsity(b)
    print(f"A Sparsity: {w_sparsity}")
    print(f"B Sparsity: {x_sparsity}")
    assert(x_sparsity >= 0 and x_sparsity < 100)
    assert(w_sparsity >= 0 and w_sparsity < 100)

    graph = dict(graph_template)
    meta_data = {
        # Operation Type
        "stonne_operation": "outerProductGEMM",

        # GEMM Parameters
        "stonne_GEMM_K": K,
        "stonne_GEMM_N": N,
        "stonne_GEMM_M": M,
        "a_hash" : hashlib.sha512(a.cpu().numpy().tobytes()).hexdigest(),
        "b_hash" : hashlib.sha512(b.cpu().numpy().tobytes()).hexdigest(),
    }
    graph[2].update(meta_data)

    # Create write path
    write_path = get_write_path(str(graph))
    os.makedirs(write_path, exist_ok=True)

    # Generating inputs
    mem_init = os.path.join(write_path, f'{prefix}_outerproduct_gemm_mem.ini')
    a_row_init = os.path.join(write_path, f'{prefix}_outerproduct_gemm_rowpointerA.in')
    a_col_init = os.path.join(write_path, f'{prefix}_outerproduct_gemm_colpointerA.in')
    b_row_init = os.path.join(write_path, f'{prefix}_outerproduct_gemm_rowpointerB.in')
    b_col_init = os.path.join(write_path, f'{prefix}_outerproduct_gemm_colpointerB.in')
    c_result = os.path.join(write_path, f'{prefix}_result.out')
    trace_path = os.path.join(write_path, "trace.py")

    if not os.path.isfile(trace_path):
        dram_a_address, dram_b_address, dram_c_address = generate_outer_product_matrix(a, b, M, K, N, prefix, write_path)
        meta_data = {
            # Memory Initialization & File Paths
            "stonne_mem_init": mem_init,
            "stonne_mem_matrix_c_file_name": c_result,

            # Memory Addresses
            "stonne_matrix_a_dram_address": dram_a_address,
            "stonne_matrix_b_dram_address": dram_b_address,
            "stonne_matrix_c_dram_address": dram_c_address,

            # CSR & Bitmap Initialization
            "stonne_rowpointer_matrix_a_init": a_row_init,
            "stonne_colpointer_matrix_a_init": a_col_init,
            "stonne_rowpointer_matrix_b_init": b_row_init,
            "stonne_colpointer_matrix_b_init": b_col_init,
            "stonne_trace_path": trace_path
        }
        graph[2].update(meta_data)

        source_code = "graph = " + str(graph)
        key, raw_tog_path = write(source_code, "py", specified_dir=write_path)
        tile_graph_generator = tog_generator(["flexagon_matmul"])
        tile_graph_generator.load_file(raw_tog_path)
        tile_graph_generator.generate_tile_graph(
            os.path.join(write_path, "tile_graph.onnx"),
            cycle_list=[0],
            x_offset=0,
            w_offset=0,
            vector_lane=0,
            stonneGraph=True
        )
        onnx_path = os.path.join(write_path, "tile_graph.onnx")
        attribute_path = os.path.join(write_path, "attributes")
        return onnx_path, attribute_path, c_result
    else: # Use trace file to generate onnx graph
        tile_graph_generator = tog_generator(["flexagon_matmul"])
        tile_graph_generator.load_file(trace_path)
        tile_graph_generator.generate_tile_graph(
            os.path.join(write_path, "trace_tile_graph.onnx"),
            cycle_list=[0],
            x_offset=0,
            w_offset=0,
            vector_lane=0,
            stonneGraph=True
        )
        onnx_path = os.path.join(write_path, "trace_tile_graph.onnx")
        attribute_path = os.path.join(write_path, "attributes")
        return onnx_path, attribute_path, c_result



def sparse_mm_stonne_outer(a, b, out):
    onnx_path, attribute_path, c_result_path = prepare_outer_product_matrix(a, b, out)

    stonne_config_path = f'{extension_config.CONFIG_TORCHSIM_DIR}/configs/stonne_single_c1_simple_noc.yml'
    result_path = TOGSimulator.run_standalone(onnx_path, config_path=stonne_config_path)
    TOGSimulator.get_result_from_file(result_path)

    # Load result data
    #with open(c_result_path, 'rb') as f:
    #    np_array = np.fromfile(f, dtype=TORCH_TO_NUMPY[out.dtype])
    #    src_tensor = torch.as_strided(torch.from_numpy(np_array), out.size(), out.stride())
    #    out.copy_(src_tensor.to(dtype=out.dtype))

def sparse_mm_dummy_stonne_outer(a, b, out):
    onnx_path, attribute_path, c_result_path = prepare_outer_product_matrix(a, b, out)
    out.copy_(torch.matmul(a.cpu(), b.cpu()))
    yield (onnx_path, attribute_path)

    # Load result data
    # with open(c_result_path, 'rb') as f:
    #     np_array = np.fromfile(f, dtype=TORCH_TO_NUMPY[out.dtype])
    #     src_tensor = torch.as_strided(torch.from_numpy(np_array), out.size(), out.stride())
    #     out.copy_(src_tensor.to(dtype=out.dtype))

custom_lib.define("_sparse_mm(Tensor a, Tensor b, Tensor out) -> Tensor")
custom_lib.impl("_sparse_mm", sparse_mm_stonne_outer, "PrivateUse1")
custom_lib.impl("_sparse_mm", sparse_mm_stonne_outer, "AutogradPrivateUse1")