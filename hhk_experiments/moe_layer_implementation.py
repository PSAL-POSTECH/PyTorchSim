import os
import sys
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Dict, Any, List
import argparse

class Experts(nn.Module):
    def __init__(
        self,
        num_local_experts: int,
        dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        dtype = torch.float16
        self.num_local_experts = num_local_experts
        self.dim = dim

        self.w1: nn.Parameter = nn.Parameter(
            torch.empty(
                dim,
                hidden_dim,
                dtype=dtype,
            )
        )

        self.w2: nn.Parameter = nn.Parameter(
            torch.empty(
                hidden_dim,
                dim,
                dtype=dtype,
            )
        )

        self.w3: nn.Parameter = nn.Parameter(
            torch.empty(
                dim,
                hidden_dim,
                dtype=dtype,
            )
        )

        # NOTE: torch.empty() leaves uninitialized values (can include NaN/Inf),
        # which will immediately propagate through matmul in fp16.
        nn.init.normal_(self.w1, mean=0.0, std=0.02)
        nn.init.normal_(self.w2, mean=0.0, std=0.02)
        nn.init.normal_(self.w3, mean=0.0, std=0.02)

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        self.prefix = prefix
        if prefix + "moe_w_in_eD_F" in state_dict:
            D = self.dim
            state_dict[prefix + "w1"] = state_dict.pop(prefix + "moe_w_in_eD_F").view(D, -1)
            state_dict[prefix + "w2"] = state_dict.pop(prefix + "moe_w_out_eF_D").view(-1, D)
            state_dict[prefix + "w3"] = state_dict.pop(prefix + "moe_w_swiglu_eD_F").view(D, -1)

    def forward(
        self,
        routed_in_egD: torch.Tensor,  # noqa: N803
    ) -> torch.Tensor:
        D = self.dim

        x_egD = routed_in_egD.view(-1, D)

        out_egD = self.batched_swiglu(x_egD, self.w1, self.w3, self.w2)
        out_egD = out_egD.view(-1, D)

        return out_egD

    def batched_swiglu(self, x: Tensor, w1: Tensor, w3: Tensor, w2: Tensor) -> Tensor:
        middle_out_egF = F.silu(torch.mm(x, w1)) * torch.mm(x, w3)
        return torch.mm(middle_out_egF, w2)


MODEL_CONFIG = {
    "LLAMA4_TP8": {
        "dim": 1024,
        "hidden_dim": 5120,
    },
    "QWEN3_TP4": {
        "dim": 384,
        "hidden_dim": 4096,
    },
}

args = argparse.ArgumentParser()

def parse_args():
    args.add_argument("--model", type=str, default="LLAMA4_TP8", help="Model to use")
    args.add_argument("--batch_size", type=int, default=1, help="Batch size")
    return args.parse_args()

if __name__ == "__main__":
    base_dir = os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim")
    sys.path.append(base_dir)
    os.environ["TOGSIM_CONFIG"] = f"{base_dir}/hhk_experiments/ndp_config.yml"
    os.environ["TORCHSIM_LOG_PATH"] = os.path.join(os.getcwd(), "togsim_results")
    args = parse_args()
    model_config = MODEL_CONFIG[args.model]
    model = Experts(num_local_experts=1, dim=model_config["dim"], hidden_dim=model_config["hidden_dim"])
    model.eval()
    model.to("npu:0")

    opt_model = torch.compile(model, dynamic=False)
    input_tensor = torch.randn(args.batch_size, model_config["dim"], dtype=torch.float16)
    output_tensor = opt_model(input_tensor.to("npu:0"))
    # print(output_tensor)
