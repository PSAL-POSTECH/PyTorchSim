# Owner(s): ["module: inductor"]
import os
import shutil
import sys
import copy
import argparse
import torch
import torch._dynamo
import torch.utils.cpp_extension
sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))
from test_transformer import EncoderBlock, test_result
from test_mlp import MLP

def apply_random_zero(tensor, zero_prob, block_size=8):
    if not 0 <= zero_prob <= 1:
        raise ValueError("zero_prob must be between 0 and 1.")

    # Generate a random mask with the same shape as the tensor
    mask = torch.rand([tensor.shape[0]//block_size, tensor.shape[1]//block_size]) > zero_prob
    mask = mask.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    # Apply the mask to the tensor (set elements to 0 where mask is False)
    return tensor * mask

def count_zeros_in_tensor_list(tensor_list):
    total_zeros = 0
    total_elements = 0
    for tensor in tensor_list:
        zeros_in_tensor = (tensor == 0).sum().item()
        total_zeros += zeros_in_tensor

        total_elements += tensor.numel()
    zero_ratio = total_zeros / total_elements if total_elements > 0 else 0
    print("Sparsity: ", zero_ratio * 100, "%")
    return total_zeros, total_elements, zero_ratio

def test_dec_inf(device, sparsity=0.0, block=8):
    torch.manual_seed(0)
    encoder_block = EncoderBlock(768, 12)
    cpu_query = torch.randn(512, 768)
    query = cpu_query.clone().to(device=device)

    cpu_y = encoder_block(cpu_query)
    with torch.no_grad():
        encoder_block.multihead_attn.linears[0].weight.copy_(apply_random_zero(encoder_block.multihead_attn.linears[0].weight, sparsity, block_size=block))
        encoder_block.multihead_attn.linears[1].weight.copy_(apply_random_zero(encoder_block.multihead_attn.linears[1].weight, sparsity, block_size=block))
        encoder_block.multihead_attn.linears[2].weight.copy_(apply_random_zero(encoder_block.multihead_attn.linears[2].weight, sparsity, block_size=block))
        encoder_block.multihead_attn.linears[3].weight.copy_(apply_random_zero(encoder_block.multihead_attn.linears[3].weight, sparsity, block_size=block))
        encoder_block.ffn1.weight.copy_(apply_random_zero(encoder_block.ffn1.weight, sparsity, block_size=block))
        encoder_block.ffn2.weight.copy_(apply_random_zero(encoder_block.ffn2.weight, sparsity, block_size=block))

    count_zeros_in_tensor_list([
        encoder_block.multihead_attn.linears[0].weight,
        encoder_block.multihead_attn.linears[1].weight,
        encoder_block.multihead_attn.linears[2].weight,
        encoder_block.multihead_attn.linears[3].weight,
        encoder_block.ffn1.weight,
        encoder_block.ffn2.weight
    ])

    encoder_block.to(device=device)
    opt_fn = torch.compile(dynamic=False)(encoder_block)
    y = opt_fn(query)
    test_result("MLP Forward", y, cpu_y)

def test_mlp_inf(device, batch_size=64, input_size=64, hidden_size=32, output_size=8, sparsity=0.0, block=8):
    torch.manual_seed(0)
    input = torch.randn(batch_size, input_size)
    x1 = copy.deepcopy(input).to(device=device)
    x2 = copy.deepcopy(input).to("cpu")
    target = torch.randn(batch_size, output_size)
    model = MLP(input_size, hidden_size, output_size)
    with torch.no_grad():
        model.linear1.weight.copy_(apply_random_zero(model.linear1.weight, sparsity, block_size=block))
        model.linear2.weight.copy_(apply_random_zero(model.linear2.weight, sparsity, block_size=block))
    count_zeros_in_tensor_list([model.linear1.weight, model.linear2.weight])
    model.requires_grad = False
    model.to(device=device)
    opt_fn = torch.compile(dynamic=False)(model)
    y = opt_fn(x1)
    cpu_model = copy.deepcopy(model).to("cpu")
    cpu_model.requires_grad = False
    cpu_y = cpu_model(x2)
    test_result("MLP Forward", y, cpu_y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count zeros in tensors from command-line arguments.")
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--block",
        type=int,
        default=8
    )
    args = parser.parse_args()

    device = torch.device("npu:0")

    #test_dec_inf(device, sparsity=args.sparsity, block=args.block)
    test_mlp_inf(device, batch_size=32, input_size=784, hidden_size=512, output_size=256, sparsity=args.sparsity, block=args.block)
