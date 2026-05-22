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


def test_expert_mask(device, batch=4, num_experts=8):
    # Regression test for issue #228:
    # (i64_buf == arange) was emitting arith.cmpi with mismatched
    # vector<Nxi64> / vector<Nxindex> operands because the operand2-side
    # index-cast branch in binary_elementwise_common was guarded by a
    # typo'd condition.
    def expert_mask(expert_idx, scores):
        j = torch.arange(num_experts, device=expert_idx.device, dtype=torch.int64)
        mask = expert_idx.unsqueeze(-1) == j.unsqueeze(0)
        return torch.where(mask, scores, torch.zeros_like(scores))

    expert_idx = torch.randint(0, num_experts, (batch,), dtype=torch.int64)
    scores = torch.randn(batch, num_experts, dtype=torch.float32)

    cpu_out = expert_mask(expert_idx, scores)

    opt_fn = torch.compile(dynamic=False)(expert_mask)
    npu_out = opt_fn(expert_idx.to(device=device), scores.to(device=device))

    test_result("ExpertMask (i64 == arange)", npu_out, cpu_out)


if __name__ == "__main__":
    device = torch.device("npu:0")
    test_expert_mask(device)
