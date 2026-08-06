"""Let Inductor's Triton codegen run on a machine with no GPU.

ONE SKEW LEFT, AND IT IS NOT A VERSION SKEW
-------------------------------------------
`triton_hash_with_backend()` asks the triton runtime driver for the *current
target*, which on a box with no GPU raises "0 active drivers". We never launch
through triton's runtime -- triton-npu compiles the kernel ahead of time to a
RISC-V ELF -- so the value is only a cache-key ingredient, and a deterministic
string does the job.

WHAT USED TO BE HERE, AND WHY IT IS GONE
----------------------------------------
On torch 2.8 there was a second, larger skew: Inductor targeted the triton ~3.3
API while triton-npu pins 3.6 (3.6 is what pins LLVM 23, and both sides of
triton-npu's textual IR seam must be the same LLVM), so `triton_key` had to be
injected back into `triton.compiler.compiler`.

torch 2.10 pins triton 3.6.0 itself and reaches that symbol through its own
compat layer (`torch._inductor.runtime.triton_compat`), so on 2.10 the versions
simply agree and the injection is a no-op. It is kept, guarded, so the module
still works if someone runs an older torch -- `_torch_handles_triton()` decides.
"""

import functools
import hashlib
import importlib
import os
import sys

_installed = False


def triton_src_dir():
    """Where tnpu's triton checkout lives (its editable install points here).

    Read out of tnpu's own `setup/versions.env` rather than guessed, so the two
    repos cannot drift: that file is the single place the checkout layout is
    pinned (TRITON_ROOT).

    TRITON_ROOT IS THE CHECKOUT, not its parent. It replaced HEXAGON_MLIR_ROOT,
    which named a directory holding triton/ and triton_shared/ side by side, so
    that one needed a "triton" path segment appended and this one must not. The
    two are still read here because a tnpu older than that rename has only the
    old key, and this repo is versioned independently of it.
    """
    from PyTorchSimFrontend import extension_config
    override = os.environ.get("TNPU_TRITON_SRC")
    if override:
        return override

    versions = os.path.join(extension_config.CONFIG_TNPU_DIR, "setup", "versions.env")
    try:
        with open(versions) as f:
            for line in f:
                if line.startswith("TRITON_ROOT="):
                    return os.path.join(line.split("=", 1)[1].strip(), "python")
                if line.startswith("HEXAGON_MLIR_ROOT="):
                    return os.path.join(
                        line.split("=", 1)[1].strip(), "triton", "python")
    except OSError:
        pass
    return "/workspace/triton-src/python"


def ensure_triton_importable():
    """`import triton` in THIS interpreter, borrowing tnpu's checkout if needed.

    Inductor's Triton codegen imports triton at codegen time (for metadata and
    hashing), so the driver needs it even though it never compiles with it.
    """
    try:
        import triton  # noqa: F401
        return True
    except ModuleNotFoundError:
        pass
    cand = triton_src_dir()
    if os.path.isdir(os.path.join(cand, "triton")):
        sys.path.insert(0, cand)
        try:
            import triton  # noqa: F401
            return True
        except ModuleNotFoundError:
            pass
    return False


def _stable_backend_hash():
    try:
        import triton
        version = triton.__version__
    except Exception:  # noqa: BLE001
        version = "unknown"
    key = f"pytorchsim-tnpu-{version}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest().upper()


def _torch_handles_triton():
    """True when this torch already knows how to reach triton's key itself.

    torch 2.10 routes it through torch._inductor.runtime.triton_compat, which
    understands triton 3.6. Older torch imports `triton_key` straight out of
    triton.compiler.compiler, where 3.6 no longer defines it.
    """
    try:
        from torch._inductor.runtime.triton_compat import triton_key  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        pass
    try:
        mod = importlib.import_module("triton.compiler.compiler")
    except Exception:  # noqa: BLE001
        return False
    return hasattr(mod, "triton_key")


def install():
    """Idempotently apply the shims. Returns a short report for logging."""
    global _installed
    notes = []
    if not ensure_triton_importable():
        raise ModuleNotFoundError(
            f"the Triton codegen route needs `triton` importable in this "
            f"interpreter (Inductor imports it during codegen). Not found, and "
            f"no checkout at {triton_src_dir()}. Set TNPU_TRITON_SRC, or install "
            f"triton into this environment.")
    if _installed:
        return notes

    if not _torch_handles_triton():
        # Pre-2.10 torch: `triton_key` is imported from triton.compiler.compiler
        # by SEVERAL call sites (codecache.CacheBase.get_system,
        # _triton.triton_hash_with_backend, ...), each with its own local import.
        # Supplying the symbol on the triton side satisfies all of them at once
        # instead of chasing every call site; it is a cache-key ingredient, so
        # any stable string will do. On torch 2.10 this branch does not run.
        mod = importlib.import_module("triton.compiler.compiler")
        mod.triton_key = _stable_backend_hash
        notes.append("injected triton.compiler.compiler.triton_key "
                     "(this torch predates the triton 3.6 compat layer)")

    # Separately: triton_hash_with_backend also asks the triton runtime driver
    # for the current target, which needs a GPU. We compile ahead of time to a
    # RISC-V ELF and never use triton's runtime, so short-circuit it.
    import torch.utils._triton as _t
    _t.triton_hash_with_backend = functools.cache(_stable_backend_hash)
    notes.append("patched torch.utils._triton.triton_hash_with_backend "
                 "(no GPU target to query)")

    _installed = True
    return notes
