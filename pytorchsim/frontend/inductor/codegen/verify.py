"""Per-kernel CPU cross-check for the functional (Spike) path.

This is a sub-option of ``pytorchsim_functional_mode``: enable it with
``TORCHSIM_FUNCTIONAL_VERIFY_PER_KERNEL=1`` (auto-disabled when functional
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
import dataclasses
import os

import torch
from torch._inductor.codegen import wrapper
from torch._inductor.virtualized import V

from pytorchsim.frontend.inductor.codegen import kernel_metadata

from pytorchsim import config
from pytorchsim.config import setup_logger

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

    Read dynamically (the config can change inside a ``with torch.npu.TOGSimulator(...)``
    block); the config accessor already AND-gates this with functional mode.
    """
    return bool(config.pytorchsim_functional_verify_per_kernel)


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
        """Every tensor output moves to CPU in its dtype, so a device-baked op or a consumer that needs
        co-located operands (aten.index) sees CPU tensors; only the recorded golden is float32. An op
        that writes its first argument and returns nothing (Inductor's reinplaced foreach) is that argument."""
        out = super().run_node(n)
        if out is None and _writes_first_argument(n.target):
            out = self.fetch_args_kwargs_from_env(n)[0][0]
        out = torch.utils._pytree.tree_map(
            lambda x: x.detach().to("cpu") if isinstance(x, torch.Tensor) else x, out)
        if isinstance(out, torch.Tensor):
            self.values[n.name] = out.to(torch.float32)
        return out


def _writes_first_argument(target):
    """Whether an aten op's schema marks its first argument as written (Tensor(a!) or Tensor(a!)[])."""
    schema = getattr(target, "_schema", None)
    alias = schema.arguments[0].alias_info if schema is not None and schema.arguments else None
    return alias is not None and alias.is_write


def _to_cpu(x):
    """A CPU copy of a tensor, never the tensor itself: the golden runs the graph's in-place ops on it."""
    return x.detach().to("cpu", copy=True) if isinstance(x, torch.Tensor) else x


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
    except Exception as e:
        logger.warning("[FuncVerify] golden computation failed (%r); "
                       "per-kernel verify disabled for this graph", e)
        _STATE["golden"] = None


def verify_check(value, buffer_name, node_name, op):
    """Compare one realized buffer against its golden; raise on first divergence. With
    TORCHSIM_FUNCTIONAL_VERIFY_TRACE every buffer's error is logged instead, as the golden is the whole
    graph on CPU and a buffer's error is all that accumulated before it."""
    if _STATE["failed"]:
        return
    golden = _STATE["golden"]
    if golden is None or not isinstance(value, torch.Tensor):
        return
    ref = golden.get(node_name)
    if ref is None:
        return
    val = value.detach().to("cpu", torch.float32)
    if val.shape != ref.shape:
        if val.numel() != ref.numel():
            return
        val = val.reshape(ref.shape)

    _STATE["n_checked"] += 1

    if os.environ.get("TORCHSIM_FUNCTIONAL_VERIFY_TRACE"):
        d = (val - ref).abs().max().item()
        scale = ref.abs().max().item()
        rel = d / scale if scale else 0.0
        logger.error("[FuncTrace] %-10s %-52s max|d|=%.4e max|ref|=%.4e rel=%.4e",
                     buffer_name, str(op)[:52], d, scale, rel)
        return

    if torch.allclose(val, ref, rtol=RTOL, atol=ATOL, equal_nan=True):
        logger.debug("[FuncVerify] PASS  %-14s %-28s %s",
                     buffer_name, op, tuple(ref.shape))
        return

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


def _writes(kernel_name, position):
    """Does `kernel_name` write the tensor argument at `position`? True when unrecorded."""
    if kernel_name is None:
        return True
    return kernel_metadata.writes_arg(kernel_name, position)


def _mutated(line):
    """Buffer names a non-kernel wrapper line (a fallback) writes in place, from Inductor's IR."""
    node = getattr(line, "node", None)
    if node is None:
        return ()
    try:
        return tuple(node.get_mutation_names())
    except NotImplementedError:
        return ()


@dataclasses.dataclass
class VerifyLine(wrapper.WrapperLine):
    """One per-kernel verify call in the wrapper body."""
    text: str

    def codegen(self, code):
        code.writeline(self.text)


class VerifyLines:
    """The wrapper's side of the cross-check: which line last writes each buffer, and the check after it."""

    def _fverify_last_writer(self):
        """{buffer name: id of the last wrapper line that writes it}.

        A buffer is checked after its last writer, since one op can span several kernels.
        """
        last = {}
        for line in self.lines:
            if isinstance(line, wrapper.KernelCallLine):
                for pos, a in enumerate(line.call_args):
                    if not (isinstance(a, str) and a.strip().isidentifier()):
                        continue
                    if not _writes(line.kernel_name, pos):
                        continue
                    last[a.strip()] = id(line)
                continue
            for name in _mutated(line):
                last[name] = id(line)
        return last

    def _fverify_emit_checks(self, call_args, line_id=None, kernel_name=None):
        """The verify_check lines after this kernel, one per buffer it writes and finishes.

        Only written arguments are checked: a reused buffer that is only read carries another origin.
        """
        checks = []
        for pos, a in enumerate(call_args):
            if not isinstance(a, str):
                continue
            name = a.strip()
            if not name.isidentifier():
                continue
            if not _writes(kernel_name, pos):
                continue
            checks += self._fverify_check_one(name, line_id)
        return checks

    def _fverify_emit_mutation_checks(self, line):
        """The same checks, after a fallback that finishes a buffer in place."""
        checks = []
        for name in _mutated(line):
            checks += self._fverify_check_one(name, id(line))
        return checks

    def _fverify_check_one(self, name, line_id):
        """The `verify_check` line for `name` ([] or one), if this line is its last writer."""
        if name in self._fverify_seen:
            return []
        if line_id is not None and self._fverify_last.get(name) != line_id:
            return []
        self._fverify_seen.add(name)
        if name in V.graph.graph_inputs:
            return []
        try:
            buf = V.graph.get_buffer(name)
        except RuntimeError:
            buf = None
        if buf is None:
            return []
        origin = getattr(buf, "origin_node", None)
        if origin is None:
            return []
        op = str(getattr(origin, "target", "?"))
        return [f'_fverify.verify_check({name}, "{name}", "{origin.name}", "{op}")']
