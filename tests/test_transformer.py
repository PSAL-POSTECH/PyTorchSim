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
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None

    def forward(self, query, key, value):
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(-1, self.h, self.d_k).transpose(0, 1)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(key, query.transpose(-2, -1)) / math.sqrt(self.d_k)
        p_attn = scores.softmax(dim=-2)
        x = torch.matmul(value.transpose(-1, -2), p_attn)
        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.view(-1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class DecoderBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DecoderBlock, self).__init__()
        self.multihead_attn = my_MultiheadAttention(num_heads, embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.ffn1 = torch.nn.Linear(embed_dim, embed_dim*4)
        self.act = torch.nn.ReLU()
        self.ffn2 = torch.nn.Linear(embed_dim*4, embed_dim)

    def forward(self, x):
        result = self.multihead_attn(x, x, x).reshape(x.shape)
        result = self.layer_norm(result+x)

        ffn1_result = self.ffn1(result)
        act_result = self.act(ffn1_result)
        ffn2_result = self.ffn2(act_result)
        return self.layer_norm(ffn2_result + result)

def test_DecoderBlock(device, head=12, embed_dim=768, input_seq=512):
    cpu_query = torch.randn(1, input_seq, embed_dim)
    decoder_block = DecoderBlock(embed_dim, head)
    cpu_res = decoder_block(cpu_query)

    query = cpu_query.clone().to(device=device)
    decoder_block.to(device=device)
    opt_fn = torch.compile(dynamic=False)(decoder_block)
    res = opt_fn(query)

    test_result("Decoder Block Forwrad", res, cpu_res)

def test_Attention(device, head=16, seq=512, d_k=64):
    def attention(query, key, value):
        import math
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = scores.softmax(dim=-1)
        return torch.matmul(p_attn, value), p_attn

    torch.manual_seed(0)
    query = torch.randn(head, seq, d_k).to(device=device)
    key = torch.randn(head, seq, d_k).to(device=device)
    value = torch.randn(head, seq, d_k).to(device=device)

    opt_fn = torch.compile(dynamic=False)(attention)
    res, p_attn = opt_fn(query, key, value)

    cpu_res, cpu_p_attn = attention(query.cpu(), key.cpu(), value.cpu())
    test_result("Attention Forward", res, cpu_res)

def test_MHA(device, num_heads=12, embed_dim=768, input_seq=512):
    MHA = my_MultiheadAttention(num_heads, embed_dim)
    cpu_query = torch.randn(input_seq, embed_dim)
    cpu_res = MHA(cpu_query, cpu_query, cpu_query)

    query = cpu_query.clone().to(device=device)
    MHA.to(device=device)
    opt_fn = torch.compile(dynamic=False)(MHA)
    res = opt_fn(query, query, query)

    test_result("MHA Forward", res, cpu_res)

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    test_DecoderBlock(device)
    # test_Attention(device, head=16, seq=512, d_k=64)
    # test_MHA(device, num_heads=12, embed_dim=768)
