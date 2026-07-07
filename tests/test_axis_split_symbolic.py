"""Unit test for symbolic-aware aligned axis splitting (axis_split.py).

Pure sympy/Inductor test (no simulator): verifies the boundary-detection and
divisibility-chain layer is a strict SUPERSET -- concrete-int reshapes behave
exactly as before, and symbolic reshapes (flattened extent E = product of dims,
divisor a genuine factor) are detected and chained correctly. The incompatible
(misaligned) and non-divisor cases must bail (no split), for both int and symbol.

Not in CI's simulator allowlist; run directly: python tests/test_axis_split_symbolic.py
"""
import sympy
from torch._inductor.utils import sympy_index_symbol
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
import PyTorchSimFrontend.mlir.axis_split as ax

v = sympy_index_symbol("v")


def I(x):
    return sympy.Integer(x)


def _chain_vals(chain):
    if chain is None:
        return None
    if all(c.is_number for c in chain):
        return [int(c) for c in chain]
    return [str(c) for c in chain]


def _boundaries(exprs, E):
    return ax.collect_boundaries(exprs, {v: 0}, {v: E}).get(0, set())


_failures = []


def check(name, got, exp):
    if got != exp:
        _failures.append(f"{name}: got {got}, expected {exp}")
        print("FAIL", name, "->", got, f"(expected {exp})")
    else:
        print("PASS", name, "->", got)


def main():
    # ---- static (must match legacy behaviour) ----
    b = _boundaries([FloorDiv(v, I(3)), ModularIndexing(v, I(1), I(3))], I(12))
    check("static reshape [4,3] boundaries", {int(x) for x in b}, {3})
    check("static reshape [4,3] chain", _chain_vals(ax._ordered_chain(b, I(12))), [1, 3, 12])

    check("static incompatible {2,3} E=6", _chain_vals(ax._ordered_chain({I(2), I(3)}, I(6))), None)

    b = _boundaries(
        [FloorDiv(v, I(12)), ModularIndexing(v, I(4), I(3)), ModularIndexing(v, I(1), I(4))],
        I(24),
    )
    check("static 3-level boundaries", {int(x) for x in b}, {4, 12})
    check("static 3-level chain", _chain_vals(ax._ordered_chain(b, I(24))), [1, 4, 12, 24])

    # ---- symbolic (new) ----
    M = sympy.Symbol("M", integer=True, positive=True)
    N = sympy.Symbol("N", integer=True, positive=True)
    A = sympy.Symbol("A", integer=True, positive=True)
    B = sympy.Symbol("B", integer=True, positive=True)
    C = sympy.Symbol("C", integer=True, positive=True)
    P = sympy.Symbol("P", integer=True, positive=True)

    b = _boundaries([FloorDiv(v, N), ModularIndexing(v, I(1), N)], M * N)
    check("sym reshape [M,N] boundaries", {str(x) for x in b}, {"N"})
    check("sym reshape [M,N] chain", _chain_vals(ax._ordered_chain(b, M * N)), ["1", "N", "M*N"])
    check("sym seg_ext M*N/N", str(ax._quotient(M * N, N)), "M")

    b = _boundaries([FloorDiv(v, B * C), ModularIndexing(v, C, B), ModularIndexing(v, I(1), C)], A * B * C)
    check("sym 3-level boundaries", {str(x) for x in b}, {"C", "B*C"})
    check("sym 3-level chain", _chain_vals(ax._ordered_chain(b, A * B * C)), ["1", "C", "B*C", "A*B*C"])

    # incomparable symbolic divisors -> bail (misaligned)
    check("sym incomparable {N,P} E=N*P", _chain_vals(ax._ordered_chain({N, P}, N * P)), None)
    # non-divisor symbolic -> no boundary collected
    check("sym non-divisor E=M*N+1", dict(ax.collect_boundaries([FloorDiv(v, N)], {v: 0}, {v: M * N + 1})), {})

    if _failures:
        raise SystemExit("Axis-split symbolic unit test FAILED:\n  " + "\n  ".join(_failures))
    print("\nAxis-split symbolic unit test: ALL PASS")


if __name__ == "__main__":
    main()
