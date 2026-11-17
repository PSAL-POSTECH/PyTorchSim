"""The spec a template choice is compiled with before its source exists: what its heuristic knows from the kernel's
inputs alone (each tensor's role and span, in the order the template's signature takes them, and the largest grid
of the tiles it offers). It reaches the choice's triton_meta through the heuristic's get_extra_kwargs."""

import json

from torch._inductor.virtualized import V

from pytorchsim.backend import compiler_input
from pytorchsim.contract import storage_span


def triton_meta_for(kernel_inputs, grids):
    """The triton_meta kwarg for a mm-family choice: nodes in signature order (an addmm's bias first), then the
    output; `grids` the (x, y, z) of each tile it is offered, the spec stating the one with the most programs."""
    tensors = [("in", _span(n.get_layout())) for n in kernel_inputs.nodes()]
    tensors.append(("out", _span(kernel_inputs.output_layout())))
    grid = [max(g[i] for g in grids) for i in range(3)]
    spec = compiler_input.spec_dict(tensors, grid, axes=("z", "y", "x"))
    return {"backend_options": compiler_input.backend_options(json.dumps(spec))}


def mm_grid(kernel_inputs, tile):
    """The mm or bmm launch grid of `tile` (block_m, block_n, ...): programs over output tiles, batches on y."""
    m, n, _ = (int(V.graph.sizevars.optimization_hint(s)) for s in kernel_inputs.mnk_symbolic())
    batch = kernel_inputs.nodes()[-1].get_size()
    batches = int(V.graph.sizevars.optimization_hint(batch[0])) if len(batch) == 3 else 1
    return (-(-m // tile[0]) * -(-n // tile[1]), batches, 1)


def _span(layout):
    hint = V.graph.sizevars.optimization_hint
    size = [int(hint(s)) for s in layout.size]
    if any(s <= 0 for s in size):
        return 0
    return storage_span(size, [int(hint(s)) for s in layout.stride], int(hint(layout.offset)))
