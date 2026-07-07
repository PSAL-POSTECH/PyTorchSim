"""Per-kernel CPU cross-check for the functional (Spike) path.

This is a sub-option of ``pytorchsim_functional_mode``: enable it with the YAML
key ``pytorchsim_functional_verify_per_kernel: 1`` (auto-disabled when functional
mode is off -- there are no Spike values to verify otherwise).

When enabled, the generated wrapper compares every realized buffer (the output of
each compiled, fused kernel that Spike just produced) against a CPU "golden"
reference. The golden is computed once per graph execution by running the
original aten graph (``V.graph.module``) on CPU with the same inputs. The first
buffer whose value diverges beyond tolerance pinpoints the fused op-cluster that
injected the error -- the finest granularity observable in a fused pipeline,
since intermediate aten ops inside a kernel are not separately materialized.

Tolerances (read once at import):
  TORCHSIM_FUNCTIONAL_VERIFY_RTOL   relative tolerance (default 1e-4)
  TORCHSIM_FUNCTIONAL_VERIFY_ATOL   absolute tolerance (default 1e-4)

The check raises FunctionalVerifyMismatch at the first divergent buffer
(stop-at-first), after logging the kernel, originating fx op, offending indices
and max abs diff.

NOTE: codegen bakes the check calls into the wrapper only when enabled at
*compile* time. Toggle the option and clear the codegen cache together
(scripts/clear_codegen_cache.sh), or a cached wrapper without checks is replayed.
"""
import os

import torch

from PyTorchSimFrontend import extension_config
from PyTorchSimFrontend.extension_config import setup_logger

logger = setup_logger(__name__)

RTOL = float(os.environ.get("TORCHSIM_FUNCTIONAL_VERIFY_RTOL", "1e-4"))
ATOL = float(os.environ.get("TORCHSIM_FUNCTIONAL_VERIFY_ATOL", "1e-4"))

# Codegen-time registry of runnable aten graphs, keyed by an int id baked into
# the generated wrapper. Persists in-process: the exec'd wrapper imports this
# very module, so the dict it sees is the one populated during codegen.
_GRAPHS = {}

# Per-run state, reset by verify_init at the top of each call(args).
_STATE = {"golden": None, "n_checked": 0, "failed": False}


class FunctionalVerifyMismatch(RuntimeError):
    """Raised at the first kernel buffer that diverges from the CPU golden."""


def enabled():
    """True when functional mode AND the per-kernel verify sub-option are on.

    Read dynamically (the config can change inside a ``with TOGSimulator(...)``
    block); the config accessor already AND-gates this with functional mode.
    """
    try:
        return bool(extension_config.pytorchsim_functional_verify_per_kernel)
    except Exception:
        return False


def register_graph(gm):
    """Register a runnable aten GraphModule, returning an id for the wrapper."""
    gid = len(_GRAPHS)
    _GRAPHS[gid] = gm
    return gid


class _GoldenInterpreter(torch.fx.Interpreter):
    """Run the aten graph on CPU, recording each node's tensor output by name."""

    def __init__(self, gm):
        super().__init__(gm)
        self.values = {}

    def run_node(self, n):
        out = super().run_node(n)
        # Move EVERY tensor output to CPU (preserving dtype) and return it, so a
        # device-baked op (e.g. arange(device=npu)) or a downstream consumer
        # (e.g. aten.index, which requires its index tensors on the indexed
        # tensor's device) sees CPU operands. Only the recorded golden is cast to
        # float32 for the allclose compare; the propagated value keeps its dtype
        # so int index tensors stay usable.
        out = torch.utils._pytree.tree_map(
            lambda x: x.detach().to("cpu") if isinstance(x, torch.Tensor) else x, out)
        if isinstance(out, torch.Tensor):
            self.values[n.name] = out.to(torch.float32)
        return out


def _to_cpu(x):
    return x.detach().cpu() if isinstance(x, torch.Tensor) else x


def verify_init(gid, inputs):
    """Compute the CPU golden for this graph execution (top of call(args))."""
    _STATE["golden"] = None
    _STATE["n_checked"] = 0
    _STATE["failed"] = False
    gm = _GRAPHS.get(gid)
    if gm is None:
        logger.warning("[FuncVerify] no graph registered for id %s", gid)
        return
    try:
        cpu_inputs = [_to_cpu(x) for x in inputs]
        interp = _GoldenInterpreter(gm)
        interp.run(*cpu_inputs)
        _STATE["golden"] = interp.values
        logger.info("[FuncVerify] golden ready: %d node tensors (rtol=%g atol=%g)",
                    len(interp.values), RTOL, ATOL)
    except Exception as e:  # never let the verify shadow break the real run
        logger.warning("[FuncVerify] golden computation failed (%r); "
                       "per-kernel verify disabled for this graph", e)
        _STATE["golden"] = None


def verify_check(value, buffer_name, node_name, op):
    """Compare one realized buffer against its golden; raise on first divergence."""
    if _STATE["failed"]:
        return
    golden = _STATE["golden"]
    if golden is None or not isinstance(value, torch.Tensor):
        return
    ref = golden.get(node_name)
    if ref is None:
        return  # no reference captured for this fx node (non-tensor / folded)
    val = value.detach().to("cpu", torch.float32)
    if val.shape != ref.shape:
        if val.numel() != ref.numel():
            return
        val = val.reshape(ref.shape)

    _STATE["n_checked"] += 1
    if torch.allclose(val, ref, rtol=RTOL, atol=ATOL, equal_nan=True):
        logger.debug("[FuncVerify] PASS  %-14s %-28s %s",
                     buffer_name, op, tuple(ref.shape))
        return

    # ---- divergence ----
    _STATE["failed"] = True
    diff = (val - ref).abs()
    tol = ATOL + RTOL * ref.abs()
    bad = diff > tol
    n_bad = int(bad.sum())
    maxd = float(diff.max())
    idxs = bad.nonzero()
    first = idxs[0].tolist() if idxs.numel() else None
    sample = [f"      {tuple(r.tolist())}: npu={val[tuple(r.tolist())].item():.6g}  "
              f"cpu={ref[tuple(r.tolist())].item():.6g}"
              for r in idxs[:6]]
    logger.error(
        "\n================= PER-KERNEL FUNCTIONAL VERIFY: DIVERGENCE =================\n"
        " first divergent buffer : %s\n"
        " originating fx op      : %s   (node '%s')\n"
        " shape                  : %s\n"
        " elements over tol      : %d / %d\n"
        " max abs diff           : %.6g   (rtol=%g atol=%g)\n"
        " first bad index        : %s\n"
        " buffers verified OK    : %d\n"
        " sample mismatches (npu vs cpu):\n%s\n"
        "===========================================================================",
        buffer_name, op, node_name, tuple(ref.shape), n_bad, ref.numel(),
        maxd, RTOL, ATOL, first, _STATE["n_checked"] - 1, "\n".join(sample))
    raise FunctionalVerifyMismatch(
        f"[FuncVerify] first divergence at buffer '{buffer_name}' (op {op}): "
        f"max abs diff {maxd:.6g}, {n_bad}/{ref.numel()} elements over tol")
