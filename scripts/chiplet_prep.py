import os
import json
import shutil
import argparse
import torch
import torch._dynamo
import torch.utils.cpp_extension

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

def test_matmul(device, input_size=128, hidden_size=128, output_size=128):
    def custom_matmul(a, b):
        return torch.matmul(a, b)
    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    x1 = input.to(device=device)
    w1 = weight.to(device=device)
    x2 = input.to("cpu")
    w2 = weight.to("cpu")
    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(x1, w1)
    y = custom_matmul(x2, w2)
    #test_result("Matmul Forward", res, y)

def modify_file(dump_path, name, address_numa_stride=None, subgraph_map=None):
    file_path = os.path.join(dump_path, 'runtime_0000', 'attribute', '0')
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return
    with open(file_path, 'r') as f:
        data = json.load(f)
    # address_numa_stride와 subgraph_map 추가
    if address_numa_stride:
        data['address_numa_stride'] = address_numa_stride
    if subgraph_map:
        data['subgraph_map'] = subgraph_map

    output_path = file_path = os.path.join(dump_path, 'runtime_0000', 'attribute')
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, name)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Modified file saved to {output_file}")

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    parser = argparse.ArgumentParser(description='Process folder argument.')
    parser.add_argument('size', type=int, help='Folder value', default=256)
    args = parser.parse_args()

    folder = int(args.size)
    print("Taget size: ", folder)
    folder_path = os.environ.get("TORCHSIM_DUMP_PATH")
    print(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    test_matmul(device, folder, folder, folder)

    pp = os.listdir(folder_path)[0]
    dump_path = os.path.join(folder_path, pp)
    pp = os.listdir(dump_path)[0]
    dump_path = os.path.join(dump_path, pp)
    subgraph_map_best = { "0": 0, "1": 0, "2": 1, "3": 1 }
    subgraph_map_worst = { "0": 1, "1": 1, "2": 0, "3": 0 }
    numa_stride = { "arg0" : [1], "arg1" : [1] , "arg2": [0 , 2] }

    subgraph_map_best1k = { "0": 0, "1": 0, "2": 1, "3": 1 }
    subgraph_map_worst1k = { "0": 1, "1": 1, "2": 0, "3": 0 }
    numa_stride_1k = { "arg0" : [1], "arg1" : [1] , "arg2": [0 , 2] }

    subgraph_map_best2k = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 1,
        "5": 1,
        "6": 1,
        "7": 1
    }
    subgraph_map_worst2k = {
        "0": 1,
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0
    }
    numa_stride_2k = { "arg0" : [2], "arg1" : [1] , "arg2": [0 , 4] }
    if args.size == 1024:
        modify_file(dump_path, "best", numa_stride_1k, subgraph_map_best1k)
        modify_file(dump_path, "worst", numa_stride_1k, subgraph_map_worst1k)
    elif args.size == 2048:
        modify_file(dump_path, "best", numa_stride_2k, subgraph_map_best2k)
        modify_file(dump_path, "worst", numa_stride_2k, subgraph_map_worst2k)
    else:
        modify_file(dump_path, "best", numa_stride, subgraph_map_best)
        modify_file(dump_path, "worst", numa_stride, subgraph_map_worst)

