"""Unit test for the masked-DMA fill's exp-reduction detection.

A log-sum-exp reduction (softmax / log_softmax) fills its masked tail with -inf so that
exp(-inf) = 0; a plain sum fills with the sum identity 0. `MLIRKernel._origin_is_exp`
decides which, and it must match the op TARGET, not a name substring -- otherwise `expand`
/ `expm1` (whose names contain "exp") would wrongly trigger the -inf fill on an ordinary
non-dividing sum, corrupting it to -inf. Skipped if torch / the frontend is unavailable.
"""
import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not available")


def test_origin_is_exp_matches_target_not_name_substring():
    import torch
    from types import SimpleNamespace as NS
    from PyTorchSimFrontend.mlir.mlir_codegen_backend import MLIRKernel
    is_exp = MLIRKernel._origin_is_exp

    # the genuine exp op -> True
    assert is_exp(NS(target=torch.ops.aten.exp.default, name="exp")) is True
    # names contain "exp" but are NOT exp -> must be False (the substring bug this guards)
    assert is_exp(NS(target=torch.ops.aten.expand.default, name="expand")) is False
    assert is_exp(NS(target=torch.ops.aten.expm1.default, name="expm1")) is False
    # unrelated op / a non-call origin (placeholder target is a str) -> False
    assert is_exp(NS(target=torch.ops.aten.log.default, name="log")) is False
    assert is_exp(NS(target="primals_1", name="primals_1")) is False
