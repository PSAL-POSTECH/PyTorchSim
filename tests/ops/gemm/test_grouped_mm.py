"""aten._grouped_mm on the npu through the Triton route: rows of A grouped by offs, group g's rows times B[g],
against a float32 CPU reference, group by group.

  routed     uneven groups (empty ones, one past BM rows) run as the grouped kernel, no call left to aten
  few_rows   fewer rows than groups: every block holds rows, the grid stops at the row count
  eager      outside torch.compile the call runs group by group (no Triton launch outside a compiled graph),
             counted as the fallback
  grad       inputs that require grad run group by group (no kernel launch), and their gradients match the CPU's
  fallback   transformers' grouped_mm_fallback (its call when the weight is not bf16) takes the same kernel

Usage: test_grouped_mm.py [case ...]   (all cases when none is named)
"""
import importlib
import sys

import torch

device = torch.device("npu:0")
TOL = 2e-2


def _inputs(sizes, k, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(sum(sizes), k, generator=g) * 0.1).to(torch.bfloat16)
    b = (torch.randn(len(sizes), k, n, generator=g) * 0.1).to(torch.bfloat16)
    offs = torch.cumsum(torch.tensor(sizes, dtype=torch.int32), 0, dtype=torch.int32)
    return a, b, offs


def _reference(a, b, offs):
    out, lo = [], 0
    for g, hi in enumerate(offs.tolist()):
        out.append(a[lo:hi].float() @ b[g].float())
        lo = hi
    return torch.cat(out)


def _gmm(a, b, offs):
    return torch.nn.functional.grouped_mm(a, b, offs=offs)


def _err(out, ref):
    return ((out.float().cpu() - ref).abs().max() / ref.abs().max()).item()


def _fallback(a, b, offs):
    return torch.ops.transformers.grouped_mm_fallback(a, b, offs=offs)


def _routed():
    from pytorchsim.frontend.dispatcher import grouped_mm
    return grouped_mm.GROUPED_MM.calls


def _run_compiled(sizes, k, n, fn=_gmm, dtype=torch.bfloat16):
    """(relative error, grouped-kernel launches, calls left to aten) of one compiled call."""
    from pytorchsim.backend import launch_hooks
    externs = []
    _routed().clear()
    launch_hooks.observe_every_extern(externs.append)
    a, b, offs = _inputs(sizes, k, n)
    a, b = a.to(dtype), b.to(dtype)
    torch._dynamo.reset()
    out = torch.compile(fn, dynamic=False)(a.to(device), b.to(device), offs.to(device))
    return _err(out, _reference(a, b, offs)), _routed().get("_grouped_mm_kernel", 0), externs


def case_routed():
    err, launched, externs = _run_compiled([0, 130, 7, 0, 64, 1, 0, 30], 256, 256)
    return err < TOL and launched == 1 and not externs, f"err {err:.3g}, launches {launched}, externs {externs}"


def case_few_rows():
    err, launched, externs = _run_compiled([1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 256, 256)
    return err < TOL and launched == 1 and not externs, f"err {err:.3g}, launches {launched}, externs {externs}"


def case_eager():
    a, b, offs = _inputs([3, 0, 9, 4], 128, 128)
    _routed().clear()
    out = _gmm(a.to(device), b.to(device), offs.to(device))
    err = _err(out, _reference(a, b, offs))
    took = dict(_routed())
    return err < TOL and took == {"fallback": 1}, f"err {err:.3g}, calls {took}"


def case_grad():
    a, b, offs = _inputs([3, 0, 9, 4], 128, 128)
    a_ref, b_ref = a.float().requires_grad_(), b.float().requires_grad_()
    _reference(a_ref, b_ref, offs).sum().backward()
    a_npu, b_npu = a.to(device).requires_grad_(), b.to(device).requires_grad_()
    _routed().clear()
    torch._dynamo.reset()
    torch.compile(_gmm, dynamic=False)(a_npu, b_npu, offs.to(device)).float().sum().backward()
    ea, eb = _err(a_npu.grad, a_ref.grad), _err(b_npu.grad, b_ref.grad)
    took = dict(_routed())
    return ea < TOL and eb < TOL and not took, f"grad err a {ea:.3g}, b {eb:.3g}, calls {took}"


def case_fallback():
    importlib.import_module("transformers.integrations.moe")
    err, launched, externs = _run_compiled([0, 130, 7, 0, 64, 1, 0, 30], 256, 256, _fallback, torch.float32)
    return err < TOL and launched == 1 and not externs, f"err {err:.3g}, launches {launched}, externs {externs}"


CASES = {"routed": case_routed, "few_rows": case_few_rows, "eager": case_eager, "grad": case_grad,
         "fallback": case_fallback}


if __name__ == "__main__":
    names = sys.argv[1:] or list(CASES)
    failed = []
    for name in names:
        ok, why = CASES[name]()
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {why}", flush=True)
        if not ok:
            failed.append(name)
    print("Test Passed" if not failed else f"Test Failed: {failed}")
    sys.exit(1 if failed else 0)
