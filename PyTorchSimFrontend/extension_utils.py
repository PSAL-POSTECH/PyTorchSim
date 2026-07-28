import sympy
import torch

"""
NOTE: Temporary File

This file contains functions that were removed or changed in newer versions
of PyTorch. It is kept here only to temporarily enable compatibility while
upgrading to PyTorch 2.8 from PyTorch 2.2.

These functions will eventually be integrated into the appropriate source files
or removed once no longer needed.

This file is not intended to be permanent and should be deleted in the future.
"""

def free_symbol_startswith(index: sympy.Expr, prefix: str):
    return any(v.name.startswith(prefix) for v in index.free_symbols)

def sympy_symbol(name: str) -> sympy.Symbol:
    # This should never be used for creating shape/stride symbols, as those
    # should all be allocated before Inductor.
    assert name[0] != "s"
    # NOTE: shape symbols are positive (> 0), but index variables are only
    # non-negative (>= 0).
    return sympy.Symbol(name, integer=True, nonnegative=True)