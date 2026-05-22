import torch
import torch.nn as nn
import torch._dynamo
import torch.utils.cpp_extension
import torch.nn.utils.prune as prune
import os
import sys
sys.path.insert(0, os.path.join(os.environ.get('TORCHSIM_DIR', default='/root/workspace/PyTorchSim'), 'tests'))
from _pytorchsim_utils import test_result

class MLP(nn.Module):
    def __init__(self, input_size=16, hidden_size=16, output_size=16, sparsity_fc1=0, sparsity_fc2=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

        prune.l1_unstructured(self.fc1, name="weight", amount=sparsity_fc1)
        prune.l1_unstructured(self.fc2, name="weight", amount=sparsity_fc2)

        prune.remove(self.fc1, "weight")
        prune.remove(self.fc2, "weight")

    def forward(self, x):
        x = torch.sparse.mm(x, self.fc1.weight)
        x = torch.sparse.mm(x, self.fc2.weight)
        return x

class SparseMLP(nn.Module):
    def __init__(self, input_size=16, hidden_size=16, output_size=16, sparsity_fc1=0, sparsity_fc2=0, device="cpu"):
        super(SparseMLP, self).__init__()

        self.weight1 = torch.empty(input_size, hidden_size, requires_grad=False)
        self.weight2 = torch.empty(hidden_size, output_size, requires_grad=False)

        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

        self._apply_pruning(self.weight1, sparsity_fc1)
        self._apply_pruning(self.weight2, sparsity_fc2)

        self.weight1 = self.weight1.to(device=device)
        self.weight2 = self.weight2.to(device=device)

        print(f"WEIGHT1 SHAPE > {self.weight1.shape}")  # (input_size, hidden_size)
        print(f"WEIGHT2 SHAPE > {self.weight2.shape}")  # (hidden_size, output_size)

    def _apply_pruning(self, tensor, sparsity):
        mask = torch.rand_like(tensor) > sparsity
        tensor *= mask

    def forward(self, x):
        x = torch.sparse.mm(x, self.weight1)
        x = torch.sparse.mm(x, self.weight2)
        return x


def test_sparse_mlp(device, batch_size=32, input_size=128, hidden_size=128, output_size=128):
    torch.manual_seed(0)
    # mlp = MLP(input_size, hidden_size, output_size)
    mlp = SparseMLP(input_size, hidden_size, output_size, device)
    mlp = mlp.to(device=device)
    input = torch.randn(batch_size, input_size)
    x1 = input.to(device=device)
    opt_fn = torch.compile(dynamic=False)(mlp)
    res = opt_fn(x1)


if __name__ == "__main__":
    import os
    import sys
    device = torch.device("npu:0")
    test_sparse_mlp(device, batch_size=8, input_size=16, hidden_size=32, output_size=64)
    
