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

def test_topk(device, size=(128, 128), k=5, dim=-1, largest=True, sorted=True):
    # dim 해석을 위해 양수 인덱스로 변환
    dim_ = dim if dim >= 0 else (len(size) + dim)
    assert 0 <= dim_ < len(size), "dim이 텐서 차원 범위를 벗어났습니다."
    assert k <= size[dim_], f"k(={k})는 size[dim](={size[dim_]}) 이하여야 합니다."

    def topk_fn(a):
        return torch.topk(a, k, dim=dim, largest=largest, sorted=sorted)

    x = torch.randn(size)
    x = x.to(device=device)

    opt_topk = torch.compile(dynamic=False)(topk_fn)
    res_values, res_indices = opt_topk(x)
    ref_values, ref_indices = torch.topk(x.cpu(), k, dim=dim, largest=largest, sorted=sorted)

    test_result("TopK/values", res_values, ref_values)
    test_result("TopK/indices", res_indices, ref_indices)

if __name__ == "__main__":
    device = torch.device('npu:0') 
    test_topk(device, (128, 128), k=2, dim=-1)