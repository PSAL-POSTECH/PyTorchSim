import torch

if __name__ == "__main__":
    device = torch.device("npu:0")
    
    # Target shape
    seq_list = [1,128,512,2048,8192]
    d_model = 768
    from tests.test_add import test_vectoradd
    from tests.test_activation import test_GeLU
    from tests.test_reduce import test_reduce_sum2
    from tests.test_layernorm import test_LayerNorm
    from tests.test_softmax import test_softmax
    func_list = [test_vectoradd, test_GeLU, test_reduce_sum2, test_LayerNorm, test_softmax]
    for test_func in func_list:
        for seq in seq_list:
            if test_func == test_GeLU:
                print(f"[log] {test_func.__name__}, seq: {seq}")
                test_func(device, size=[seq, d_model*4])
            elif test_func == test_softmax:
                print(f"[log] {test_func.__name__}, seq: {seq}")
                test_func(device, size=[seq, seq])
            else:
                print(f"[log] {test_func.__name__}, seq: {seq}")
                test_func(device, size=[seq, d_model])
