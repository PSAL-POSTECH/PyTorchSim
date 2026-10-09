"""Torch's triton_helpers, read from torch as source for the torch-free compiler venv.

A kernel's `triton_helpers.X` calls are served by a module object over the @triton.jit
helpers it uses, their dependencies included, renamed `_torchsim_*` and prepended.
"""

import inspect
import re

from .errors import SpecIncomplete

_PREAMBLE = """import types as _types
import math as _torchsim_pymath
import triton
import triton.language as tl
from triton.language import math
from triton.language.extra import libdevice
from triton.language.standard import _log2
_LOG_2_E: tl.constexpr = tl.constexpr(_torchsim_pymath.log2(_torchsim_pymath.e))
"""
_PREAMBLE_NAMES = {"triton", "tl", "math", "libdevice", "_log2", "_LOG_2_E", "pymath"}


def _module():
    """torch._inductor.runtime.triton_helpers."""
    from torch._inductor.runtime import triton_helpers
    return triton_helpers


def _jit_helpers(mod):
    """{name: source} of every @triton.jit function the module itself defines."""
    out = {}
    for name, obj in vars(mod).items():
        fn = getattr(obj, "fn", None)
        if fn is not None and getattr(fn, "__module__", None) == mod.__name__:
            out[name] = inspect.getsource(fn)
    return out


def _names(src):
    """Bare (non-attribute) identifiers a source refers to."""
    return set(re.findall(r"(?<![\w.])([A-Za-z_]\w*)\b", src))


def helper_source(used):
    """Source defining the `triton_helpers` module object for the helpers `used`.

    Raises SpecIncomplete for a name that is no @triton.jit helper or needs a global this cannot supply.
    """
    mod = _module()
    helpers = _jit_helpers(mod)
    missing = sorted(set(used) - set(helpers))
    if missing:
        raise SpecIncomplete(
            f"kernel uses triton_helpers.{{{','.join(missing)}}}, which is not a "
            f"@triton.jit helper in torch's triton_helpers")

    need, todo = set(), list(used)
    while todo:
        name = todo.pop()
        if name in need:
            continue
        need.add(name)
        todo.extend(n for n in _names(helpers[name]) if n in helpers and n not in need)

    module_globals = set(vars(mod)) - set(helpers) - _PREAMBLE_NAMES
    for name in sorted(need):
        unknown = sorted(_names(helpers[name]) & module_globals)
        if unknown:
            raise SpecIncomplete(
                f"triton_helpers.{name} needs {unknown} from torch, which the "
                f"compiler venv has no copy of")

    rename = re.compile(r"(?<![\w.])(" + "|".join(sorted(need, key=len, reverse=True)) + r")\b")
    body = "\n".join(rename.sub(r"_torchsim_\1", helpers[n].replace("pymath.", "_torchsim_pymath."))
                     for n in sorted(need))
    binds = "\n".join(f"triton_helpers.{n} = _torchsim_{n}" for n in sorted(used))
    return (f"\n{_PREAMBLE}\n{body}\n"
            f"triton_helpers = _types.ModuleType('triton_helpers')\n{binds}\n")
