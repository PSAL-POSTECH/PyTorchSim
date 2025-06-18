import math
import copy
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

def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class my_MultiheadAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(my_MultiheadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linear = torch.nn.Linear(d_model, d_model)
        self.attn = None

    def forward(self, query, key, value):
        # BMM + Max
        scores = torch.matmul(key, query.transpose(-2, -1))
        s_max = scores.max(dim=-2, keepdim=True).values

        # Reduce Sum
        scores = torch.exp(scores-s_max)
        s_sum = scores.sum(dim=-2, keepdim=True)

        # Elementwise + BMM
        p_attn = scores/s_sum
        x = torch.matmul(value.transpose(-1, -2), p_attn)
        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.view(-1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linear(x)

def test_MHA(device, num_heads=12, embed_dim=768, input_seq=512):
    MHA = my_MultiheadAttention(num_heads, embed_dim)
    cpu_query = torch.randn(num_heads, input_seq, embed_dim//num_heads)
    cpu_key = torch.randn(num_heads, input_seq, embed_dim//num_heads)
    cpu_value = torch.randn(num_heads, input_seq, embed_dim//num_heads)
    cpu_res = MHA(cpu_query, cpu_key, cpu_value)

    query = cpu_query.clone().to(device=device)
    key = cpu_key.clone().to(device=device)
    value = cpu_value.clone().to(device=device)
    MHA.to(device=device)
    opt_fn = torch.compile(dynamic=False)(MHA)
    res = opt_fn(query, key, value)

    test_result("MHA Forward", res, cpu_res)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_MHA(device)
    # test_Attention(device, head=16, seq=512, d_k=64)
    # test_MHA(device, num_heads=12, embed_dim=768)
