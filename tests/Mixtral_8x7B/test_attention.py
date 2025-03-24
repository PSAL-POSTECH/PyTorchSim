import copy
import torch
import torch._dynamo
import torch.utils.cpp_extension
from model import Transformer, TransformerBlock, ModelArgs, Attention, FeedForward, KVCache, precompute_freqs_cis, sample

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    message = f"|{name} Test Passed|"
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

def test_decode(device, prompt_length, nr_tokens):
    # Setup model & model args
    args = ModelArgs()
    args.n_head = 8
    args.n_local_heads = -1
    args.intermediate_size = None
    args.dim = 512
    args.n_layer = 1
    args.__post_init__()
    max_batch = 1
    max_seq = 512
    head_dim = args.dim // args.n_head
    model = Transformer(args)
    model.setup_caches(max_batch, max_seq)
    model = model.to(device=device)

    # Prepare inputs
    T = prompt_length
    prompt = torch.randn([1, T, args.dim] , dtype=torch.float32)
    cpu_prompt = copy.deepcopy(prompt)
    cpu_model = copy.deepcopy(model).to("cpu")
    opt_fn = torch.compile(dynamic=False)(model)

    # Prepare KV cache
    kv_caches = [KVCache(max_batch, max_seq, args.n_head, head_dim, torch.float32) for i in range(args.n_layer)]
    cpu_kv_caches = copy.deepcopy(kv_caches)
    kv_caches = [kv.to(device=device) for kv in kv_caches]

    for i in range(nr_tokens):
        input_pos = torch.arange(0, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        freqs_cis = precompute_freqs_cis(args.block_size, args.dim // args.n_head, args.rope_base)[input_pos].to(dtype=torch.float32)
        prompt = prompt.to(device=device)
        cpu_input_pos = copy.deepcopy(input_pos)
        input_pos = input_pos.to(device=device)
        cpu_mask = copy.deepcopy(mask)
        mask = mask.to(device=device)

        freqs_cis = freqs_cis.view(1, T, 1, -1)
        cpu_freqs_cis = copy.deepcopy(freqs_cis)
        freqs_cis = freqs_cis.to(device=device)

        # Run models
        res = opt_fn(prompt, mask, freqs_cis, input_pos, kv_caches)
        cpu_res = cpu_model(cpu_prompt, cpu_mask, cpu_freqs_cis, cpu_input_pos, cpu_kv_caches)
        new_token = sample(cpu_res.cpu())[0]
        print(new_token)
        new_token = cpu_model.tok_embeddings(new_token).unsqueeze(1)
        cpu_prompt = new_token #torch.cat([cpu_prompt, new_token], dim=1)
        prompt = cpu_prompt.clone()
        T = 1

        # Check output token
        test_result("Mistral", res, cpu_res)
        print("Max diff > ", torch.max(torch.abs(res.cpu() - cpu_res)))

def test_attention(device):
    args = ModelArgs()
    args.n_head = 8
    args.n_local_heads = -1
    args.intermediate_size = None
    args.dim = 512
    args.__post_init__()
    model = Attention(args)
    model = model.to(device=device)

    T = 32
    prompt = torch.randn([1, T, args.dim] , dtype=torch.float32)
    input_pos = torch.arange(0, T)
    cpu_prompt = copy.deepcopy(prompt)
    prompt = prompt.to(device=device)
    cpu_input_pos = copy.deepcopy(input_pos)
    input_pos = input_pos.to(device=device)

    freqs_cis = precompute_freqs_cis(args.block_size, args.dim // args.n_head, args.rope_base)[cpu_input_pos].to(dtype=torch.float32)
    freqs_cis = freqs_cis.view(1, T, 1, -1)
    cpu_freqs_cis = copy.deepcopy(freqs_cis)
    freqs_cis = freqs_cis.to(device=device)

    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    cpu_mask = copy.deepcopy(mask)
    mask = mask.to(device=device)

    cpu_model = copy.deepcopy(model).to("cpu")
    opt_fn = torch.compile(dynamic=False)(model)
    res = opt_fn(prompt, freqs_cis, mask, input_pos)
    cpu_res = cpu_model(cpu_prompt, cpu_freqs_cis, cpu_mask, cpu_input_pos)

    test_result("Attention", res, cpu_res)
    print("Max diff > ", torch.max(torch.abs(res.cpu() - cpu_res)))

def test_ffn(device):
    args = ModelArgs()
    args.n_head = 8
    args.n_local_heads = -1
    args.intermediate_size = None
    args.dim = 512
    args.__post_init__()
    model = FeedForward(args)
    model = model.to(device=device)

    T = 32
    prompt = torch.randn([1, T, args.dim] , dtype=torch.float32)
    cpu_prompt = copy.deepcopy(prompt)
    prompt = prompt.to(device=device)

    cpu_model = copy.deepcopy(model).to("cpu")
    opt_fn = torch.compile(dynamic=False)(model)
    res = opt_fn(prompt)
    cpu_res = cpu_model(cpu_prompt)
    test_result("FFN", res, cpu_res)

def test_concat(device, size1=(1, 8, 32, 64), size2=(1, 8, 1, 64), dim=2):
    def concat_tensors(a, b):
        return torch.cat((a, b), dim=dim)

    x = torch.randn(size1)
    y = torch.randn(size2)
    cpu_x = x.clone()
    cpu_y = y.clone()
    x = x.to(device=device)
    y = y.to(device=device)

    opt_fn = torch.compile(dynamic=False)(concat_tensors)
    res = opt_fn(x, y)
    out = concat_tensors(cpu_x, cpu_y)

    test_result("ConcatTensors", res, out)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_decode(device, 32, 3)
    # test_concat(device, size1=(1, 8, 32, 64), size2=(1,8,1,64), dim=2)
    # test_attention(device)
    # test_ffn(device)
