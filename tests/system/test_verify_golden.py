"""The per-kernel verify's CPU golden runs the graph on copies of its inputs, and an op Inductor
reinplaced (an in-place foreach that returns nothing, read through getitem) is the argument it
wrote. The graph is built by hand, so no device is needed."""
import operator
import os
import sys

sys.path.insert(0, os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"))

import torch  # noqa: E402
from torch import fx  # noqa: E402

from pytorchsim.frontend.inductor.codegen import verify  # noqa: E402

aten = torch.ops.aten


def _reinplaced_foreach():
    """(a, b) -> _foreach_add_([a, b], 1); the sum of its getitems, as Inductor leaves Adam's step."""
    g = fx.Graph()
    a, b = g.placeholder("a"), g.placeholder("b")
    added = g.call_function(aten._foreach_add_.Scalar, ([a, b], 1))
    first, second = (g.call_function(operator.getitem, (added, i)) for i in (0, 1))
    g.output(g.call_function(aten.add.Tensor, (first, second)))
    return fx.GraphModule(torch.nn.Module(), g)


def test_golden_of_a_reinplaced_foreach_leaves_the_inputs():
    a, b = torch.zeros(4), torch.ones(4)
    verify.verify_init(verify.register_graph(_reinplaced_foreach()), [a, b])
    golden = verify._STATE["golden"]
    assert golden is not None, "the golden was not computed"
    assert torch.equal(golden["add_tensor"], torch.full((4,), 3.0)), golden["add_tensor"]
    assert torch.equal(a, torch.zeros(4)) and torch.equal(b, torch.ones(4)), (a, b)


if __name__ == "__main__":
    test_golden_of_a_reinplaced_foreach_leaves_the_inputs()
    print("PASS")
