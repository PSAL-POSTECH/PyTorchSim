# Adding a post-grad rewrite

[한국어](post_grad_rewrite.ko.md)

A post-grad rewrite replaces one op in an npu graph with other aten ops that Inductor can lower, so
the work stays on the device. Without it, an op with no npu kernel runs as an extern aten call on
the CPU, outside the simulator. histc, a 1 x K row times a matrix, transposed convolution,
zero-length matmuls and complex arithmetic are rewrites. They all live in `pytorchsim/frontend/inductor/post_grad_pass.py`. A rewrite of one op is
one decorated function, so you write the pattern, when it applies and the replacement, nothing else.

## First, check that you need one

| The op | What to do |
|---|---|
| Inductor lowers it, and PyTorchSim-Triton-Backend compiles the Triton it generates | Nothing. Most ops are this. |
| It has no npu kernel, and other aten ops can express it (histc, polar, complex arithmetic) | A rewrite: this guide |
| Inductor's kernels for it are not the kernel the hardware runs (SDPA as one fused kernel, MoE's grouped GEMM, topk in registers) | A fused kernel: [fused_kernel.md](fused_kernel.md) |

Trace the op first and look at what reaches the pass. The pass sees the graph after Inductor's
decomposition table, so some ops never arrive: `torch.special.ndtr`, for example, is already
`erf` and arithmetic by then.

## How a node is handled

`registration.py` registers `pytorchsim_post_grad_pass` as the `device_custom_pass` of the npu
(`register_backend_for_device`), so Inductor runs it on npu graphs only. It runs the graph patterns
(`PATTERNS.apply(gm)`), then `rewrite_complex_as_real_pairs`. Every node takes one of these routes:

```
npu graph, after Inductor's decompositions
 ├─ PATTERNS.apply(gm), for each node whose target a pattern names
 │   ├─ a complex input or output                   → skipped by Inductor's matcher
 │   ├─ the pattern does not match,                 → left alone            not counted
 │   │  or extra_check(match) is False
 │   └─ fn(node, *args, **kwargs) returns
 │       ├─ (replacement, example_args)             → match.replace_by_example   counted "rewritten"
 │       └─ Decline("why")                          → the node stays             logged, counted "fallback"
 └─ rewrite_complex_as_real_pairs, for each group of complex nodes joined by complex values
     ├─ every op has a _REAL_PAIR rule              → real [..., 2] pairs    "real_pairs": "rewritten"
     └─ otherwise                                   → the group stays        logged, "real_pairs": "fallback"
```

A node that stays runs where `fallback_route(target)` says, and the log line names it: Inductor's
own kernel, or an extern aten call on the CPU that the simulator does not see. The pass traces the
replacements under the graph's own fake mode, so it also runs on a graph a test hands it directly.

## The pieces

```python
@npu_rewrite(CallFunction(aten.my_op.default, Arg(), Arg()), extra_check)  # the nodes it takes
def my_op(node, x, dim):                                 # the matched node and its arguments
    def compute(x, *, dim): ...                          # the replacement, plain torch code
    if _val(x) is None:
        return Decline("its input has no fake value")    # the node stays, with the reason
    return functools.partial(compute, dim=dim), [x]      # the replacement and its tensor arguments
```

| Piece | What it is |
|---|---|
| `npu_rewrite(pattern, extra_check)` | The decorator. It registers the function in `PATTERNS` (an Inductor `PatternMatcherPass`) and counts each result in `CALLS` under the function's name. `extra_check` defaults to taking every match. Called as a plain function, it registers one function for several patterns (`zero_length_gemm`). |
| `pattern` | Inductor's pattern language from `torch._inductor.pattern_matcher`. `CallFunction(op, Arg(), ...)` names the op and one `Arg()` per argument. `CallFunctionVarArgs(op)` takes any number of arguments, for an op whose optional arguments may be left out (histc). |
| `extra_check(match)` | Picks the nodes the rewrite applies to. `False` means the node is not this rewrite's business: nothing is logged or counted. `match.args` are the captured arguments; read their fake values with `_val`. |
| `fn(node, *args, **kwargs)` | Gets the matched node and its arguments. It returns `(replacement, example_args)`: `replacement` is plain torch code that Inductor traces in the node's place (`match.replace_by_example`), and `example_args` are the graph nodes it takes. Close over constants with `functools.partial`, so only tensor nodes are example arguments. |
| `Decline("reason")` | The other return: the rewrite cannot take this node. The node stays (the fallback), and the log line gives the reason and where the node runs instead. It is the same `Decline` a fused kernel's `plan` returns. |
| `CALLS` | `{rewrite name: {"rewritten": n, "fallback": n}}`, counted per node. Complex values count under `"real_pairs"`, once per group. |
| `_REAL_PAIR` | `{complex op: rule}`. A rule writes one complex op on real `[..., 2]` pairs. It has rules for `aten.polar`, `view_as_complex`, `unsqueeze` and `mul` (`Tensor` and `Scalar`). |

Complex ops are not patterns. Inductor's matcher skips a node with a complex input or output
(`fallback_node_due_to_unsupported_type`), so a pattern on a complex op never fires. The real-pair
pass takes a whole group of complex nodes or none of it: one op without a rule leaves the group as
it is.

## Step by step

### 1. One function in `post_grad_pass.py`

Put it after the existing rewrites (`histc`, `row_times_matrix`, `transposed_conv`,
`zero_length_gemm`), before `rewrite_complex_as_real_pairs`. The replacement goes inside the rewrite,
as `histc`'s `counts` does, and the checks and the return follow it. `histc` is the shortest real
example. Here is a new one: `aten.logcumsumexp`, which Inductor lowers only as a fallback, so on the
npu it runs on the CPU. Each output is a logsumexp over a prefix, and `logsumexp` lowers.

```python
@npu_rewrite(CallFunction(aten.logcumsumexp.default, Arg(), Arg()))
def logcumsumexp(node, x, dim):
    """aten.logcumsumexp as one logsumexp per prefix (prefix_logsumexp)."""

    def prefix_logsumexp(x, *, dim):
        """Output i is the logsumexp of x[..., :i+1] along dim: x against the prefix mask, the rest -inf, so
        each prefix takes its own max (one shift by the row max sends early, small terms to 0)."""
        n = x.shape[dim]
        x = x.movedim(dim, -1)
        i = torch.arange(n, device=x.device)
        prefix = i.reshape(-1, 1) >= i.reshape(1, -1)
        y = torch.where(prefix, x.unsqueeze(-2), float("-inf")).logsumexp(-1)
        return y.movedim(-1, dim)

    v = _val(x)
    if v is None:
        return Decline("its input has no fake value")
    if not v.is_floating_point():
        return Decline(f"{v.dtype} is not a floating dtype")
    return functools.partial(prefix_logsumexp, dim=dim), [x]
```

The integral input is declined because aten has no integral logcumsumexp: the node stays and raises
as aten does. The replacement does n² work per row where aten does n. It is vector work on the
device instead of a CPU call the simulator does not see, so say so in the PR.

### 2. A complex op: one rule in `_REAL_PAIR`

A rule gets `call` (it inserts a node before the complex one and sets its fake value), the complex
node `n`, and `pair` (each complex input's real pair so far). It returns the node holding `n`'s
pair. For `-z`:

```python
def _pair_of_neg(call, n, pair):
    """-z: the pair negated."""
    return call(aten.neg.default, pair[n.args[0]])


_REAL_PAIR = {
    ...
    aten.neg.default: _pair_of_neg,
}
```

A target in `_REAL_PAIR` counts as covered for every call of it, so a rule must handle every form
the op takes. `_pair_of_mul` handles complex times complex, complex times real and a Python complex
scalar. The nodes a rule inserts are not decomposed again, so each one must have an Inductor
lowering as it is.

### 3. Test the graph and the route, not only the values

Without the rewrite, the node falls back and still gives the right answer on the CPU. So a test that
only compares values still passes when the rewrite never fires. Check the graph and `CALLS` as well.
Copy the shape of [test_complex_to_real.py](../tests/ops/misc/test_complex_to_real.py): trace the
function as Inductor hands it to its post-grad passes, run the pass, and check what is left.

```python
gm = make_fx(f, decomposition_table=select_decomp_table(), tracing_mode="fake")(x)
CALLS.clear()
pytorchsim_post_grad_pass(gm)
assert not [n for n in gm.graph.nodes if n.target is aten.logcumsumexp.default]
assert not _unlowered(gm), f"no Inductor lowering for {_unlowered(gm)}"
check_result("logcumsumexp (rewritten graph)", gm(x), f(x))
assert CALLS == {"logcumsumexp": {"rewritten": 1}}, CALLS
```

`_unlowered(gm)` lists the ops Inductor has no lowering for. Copy it from either test file. When
`make_fx` cannot trace the op on the CPU, build the one-node graph by hand with fake values, as
`_histc_graph` in [test_histc.py](../tests/ops/misc/test_histc.py) does. Give each case the route it
must take:

| Case | Expected `CALLS` |
|---|---|
| a traced graph with one call | `{"logcumsumexp": {"rewritten": 1}}`. The op is gone, `_unlowered(gm)` is empty, the values match. |
| two calls in one graph | `{"logcumsumexp": {"rewritten": 2}}` (nodes are counted) |
| a hand-built node whose input has no fake value | `{"logcumsumexp": {"fallback": 1}}`. The node stays. |
| an integral input | `{"logcumsumexp": {"fallback": 1}}`. The node stays. |
| the edge cases: `-inf` prefixes, `[-200, 0]` in float32, both signs of `dim` | rewritten, values equal to aten's |
| a complex group with the new rule | `{"real_pairs": {"rewritten": 1}}`, no complex node left |
| a complex group with an op without a rule | `{"real_pairs": {"fallback": 1}}`, the group unchanged |
| `torch.compile(f)` on `npu:0` | the values match the CPU's |

Clear `CALLS` before each case, and call `torch._dynamo.reset()` before each compiled one. The pass
runs once per compiled graph, so `CALLS` counts compiles, not runs. The end-to-end case checks that
the rewrite gives the right answer once real kernels compute it; the graph cases are what pin the
rewrite.

Run it on the npu as a script: `python tests/ops/misc/test_<op>.py`.

## Rules

These follow the maintainer's review rules for this repo.

- **Judge nodes by target.** Compare `n.target is aten.x.default`, or check `callable(n.target)`.
  Do not compare op strings such as `node.op == "call_function"`, in the pass or in tests.
- **Do not parse text.** Read fake values (`_val`), targets and arguments. Never read a printed
  graph or kernel source.
- **No silent fallback.** When `fn` cannot take a node, return `Decline` with the reason: the
  reason is the log line. Use `extra_check` only for nodes that are not this rewrite's business.
- **No shape special cases.** Do not pick nodes by particular shapes or row counts, and do not add
  a later exception that undoes one. `_is_row_times_matrix` takes the general form (any 1 x K row).
- **Derive constants from the machine.** Take sizes from the fake values or
  `config.CONFIG_MACHINE`. Do not hard-code them.
- **The replacement must match the op**: the same values, dtypes and edge-case rules. `histc` keeps
  aten's top edge in the last bin, drops values outside `[min, max]`, widens `min == max`, and
  returns the input's dtype. Test each such rule.
- **Use public PyTorch where one exists.** `torch._inductor.pattern_matcher` is the mechanism
  Inductor's own post-grad passes use, and the pass sets the graph's fake mode for tracing through
  `torch._inductor.virtualized.V.set_fake_mode`. Neither has a public counterpart, so a PR that
  adds a rewrite must say so.
- **Keep rewrites in `post_grad_pass.py`.** The pass's `uuid` hashes that file, so an edit there is
  a new cache key. A rewrite in another file would need adding to the hash, or a compile could
  reuse a graph built without it.

## Extra patterns

- **Optional arguments**: use `CallFunctionVarArgs(op)` and fill the defaults in `fn`, as `histc`
  does with `bins`, `min` and `max`.
- **One check for several ops**: `row_times_matrix` (`aten.mm`) and `row_times_matrix_plus`
  (`aten.addmm`) share `_is_row_times_matrix`, which reads the last two arguments of either.
- **One rewrite for several ops**: call `npu_rewrite(pattern, check)(fn)` in a loop.
  `zero_length_gemm` is registered for `mm`, `bmm`, `addmm` and `baddbmm` this way, with the check
  built per op (`_is_zero_length_gemm(bias)`), and counts under one name in `CALLS`.
- **A check on a constant argument**: `transposed_conv` takes `aten.convolution` only when its
  `transposed` argument is `True` (`match.args[6] is True`).
- **One `Arg()` per argument.** The matcher reads one pattern object used twice as "the same value
  twice", so `CallFunction(op, *[Arg()] * 9)` matches nothing ("repeated pattern differs"). Write
  `CallFunction(op, *(Arg() for _ in range(9)))`, as `transposed_conv` does.
- **Constants in the replacement**: close over them with `functools.partial`. Only graph nodes go
  in `example_args`.
