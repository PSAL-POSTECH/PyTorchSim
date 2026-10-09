"""Let Inductor's Triton codegen run on a machine with no GPU.

`triton_hash_with_backend()` asks the runtime driver for the current target,
which needs a GPU; we compile ahead of time to a RISC-V ELF and never use
triton's runtime, so the value is only a cache-key ingredient.
"""

import functools
import hashlib
import importlib
import os
import sys

_installed = False


def triton_src_dir():
    """Where the compiler's triton checkout lives (its editable install points here).

    Read out of the compiler's setup/versions.env, where TRITON_ROOT is the checkout.
    """
    from PyTorchSimFrontend import config
    override = os.environ.get("TORCHSIM_COMPILE_TRITON_SRC")
    if override:
        return override

    versions = os.path.join(config.CONFIG_TORCHSIM_COMPILE_DIR, "setup", "versions.env")
    try:
        with open(versions) as f:
            for line in f:
                if line.startswith("TRITON_ROOT="):
                    return os.path.join(line.split("=", 1)[1].strip(), "python")
    except OSError:
        pass
    return "/workspace/triton-src/python"


def ensure_triton_importable():
    """`import triton` in THIS interpreter, borrowing the compiler's checkout if needed.

    Inductor's Triton codegen imports triton at codegen time for metadata and
    hashing, so the driver needs it even though it never compiles with it.
    """
    try:
        import triton
        return True
    except ModuleNotFoundError:
        pass
    cand = triton_src_dir()
    if os.path.isdir(os.path.join(cand, "triton")):
        sys.path.insert(0, cand)
        try:
            import triton
            return True
        except ModuleNotFoundError:
            pass
    return False


def _stable_backend_hash():
    try:
        import triton
        version = triton.__version__
    except Exception:
        version = "unknown"
    key = f"pytorchsim-torchsim-compile-{version}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest().upper()


def _torch_handles_triton():
    """True when this torch already knows how to reach triton's key itself.

    torch 2.10 routes it through triton_compat, which understands triton 3.6;
    older torch imports triton_key straight out of triton.compiler.compiler.
    """
    try:
        from torch._inductor.runtime.triton_compat import triton_key
        return True
    except Exception:
        pass
    try:
        mod = importlib.import_module("triton.compiler.compiler")
    except Exception:
        return False
    return hasattr(mod, "triton_key")


def install():
    """Idempotently apply the shims.

    Supplying triton_key on the triton side satisfies every pre-2.10 call site
    at once; on 2.10 that branch does not run.
    """
    global _installed
    if not ensure_triton_importable():
        raise ModuleNotFoundError(
            f"the Triton codegen route needs `triton` importable in this "
            f"interpreter (Inductor imports it during codegen). Not found, and "
            f"no checkout at {triton_src_dir()}. Set TORCHSIM_COMPILE_TRITON_SRC, or install "
            f"triton into this environment.")
    if _installed:
        return

    if not _torch_handles_triton():
        mod = importlib.import_module("triton.compiler.compiler")
        mod.triton_key = _stable_backend_hash

    import torch.utils._triton as _t
    _t.triton_hash_with_backend = functools.cache(_stable_backend_hash)

    _installed = True
