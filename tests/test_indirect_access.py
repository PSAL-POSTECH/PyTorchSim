import torch
import copy

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

def test_indirect_vectoradd(device, size=(128, 128)):
    def vectoradd(a, idx, b):
        return a[idx] + b
    x = torch.randn(size, dtype=torch.float32).to(device=device)
    idx = torch.randint(0,128, [128]).to(device=device)
    y = torch.randn(128, dtype=torch.float32).to(device=device)
    opt_fn = torch.compile(dynamic=False)(vectoradd)
    res = opt_fn(x, idx, y)
    out = vectoradd(x.cpu(), idx.cpu(), y.cpu())
    test_result("Indirect VectorAdd", res, out)

def test_embedding(device, vocab_size, dim):
    emb = torch.nn.Embedding(vocab_size, dim)
    cpu_emb = copy.deepcopy(emb)

    prompt = torch.randint(0, 1023, [511], dtype=torch.int)
    cpu_prompt = copy.deepcopy(prompt)
    prompt = prompt.to(device=device)

    emb.to(device=device)
    opt_emb = torch.compile(dynamic=False)(emb)
    res = opt_emb(prompt)
    cpu_res = cpu_emb(cpu_prompt)
    test_result("Embedding", res, cpu_res)

def test_scatter_add(device, num_tokens=256, hidden_size=256, num_assignments=3, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)

    def scatter_only(out, token_indices, weighted_output):
        # token_indices: [N] (long), weighted_output: [N, H]
        out.index_add_(0, token_indices, weighted_output)
        return out

    out = torch.randn(num_tokens, hidden_size, dtype=dtype)
    out_cp = out.clone()
    token_indices = torch.randint(0, num_tokens, (num_assignments,))
    weighted_output = torch.randn(num_assignments, hidden_size, dtype=dtype)

    cpu_out = scatter_only(out, token_indices, weighted_output)

    out = out_cp.to(device=device)
    token_indices = token_indices.to(device=device)
    weighted_output = weighted_output.to(device=device)
    opt_fn = torch.compile(dynamic=False)(scatter_only)
    res = opt_fn(out, token_indices, weighted_output)
    test_result("ScatterAdd(index_add_)", res, cpu_out)

def test_scatter_full(device, size=(128, 128)):
    def vectoradd(a, idx, b):
        a[idx, :] = b
        return a
    x = torch.randn(size, dtype=torch.float32).to(device=device)
    x_cpu = x.clone().cpu()
    idx = torch.randint(0,128, [128]).to(device=device)
    y = torch.randn(size[1], dtype=torch.float32).to(device=device)
    opt_fn = torch.compile(dynamic=False)(vectoradd)
    res = opt_fn(x, idx, y)
    out = vectoradd(x_cpu, idx.cpu(), y.cpu())
    test_result("Indirect VectorAdd", res, out)

if __name__ == "__main__":
    device = torch.device("npu:0")
    test_scatter_full(device)
    test_scatter_full(device, size=(2048, 2048))
    test_scatter_add(device)
    test_indirect_vectoradd(device)
    #test_embedding(device, 1024, 2048)