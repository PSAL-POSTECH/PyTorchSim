"""Let Inductor's Triton codegen run on a machine with no GPU.

`triton_hash_with_backend()` asks the triton runtime driver for the current
target, which raises "0 active drivers" without a GPU. We compile ahead of time
to a RISC-V ELF and never launch through that runtime, so the value is only a
cache-key ingredient and a deterministic string does.

Pre-2.10 torch also needs `triton_key` injected into `triton.compiler.compiler`;
2.10 pins triton 3.6 itself and reaches it through its own compat layer.
`_torch_handles_triton()` decides.
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
    pinned (HEXAGON_MLIR_ROOT).
    """
    from PyTorchSimFrontend import extension_config
    override = os.environ.get("TNPU_TRITON_SRC")
    if override:
        return override

    root = "/workspace/hexagon-mlir"
    versions = os.path.join(extension_config.CONFIG_TNPU_DIR, "setup", "versions.env")
    try:
        with open(versions) as f:
            for line in f:
                if line.startswith("HEXAGON_MLIR_ROOT="):
                    root = line.split("=", 1)[1].strip()
                    break
    except OSError:
        pass
    return os.path.join(root, "triton", "python")


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
        # Several call sites import triton_key with their own local import;
        # supplying it on the triton side satisfies all of them at once.
        mod = importlib.import_module("triton.compiler.compiler")
        mod.triton_key = _stable_backend_hash
        notes.append("injected triton.compiler.compiler.triton_key "
                     "(this torch predates the triton 3.6 compat layer)")

    # triton_hash_with_backend asks the runtime driver for the current target,
    # which needs a GPU. Short-circuited; it is only a cache key.
    import torch.utils._triton as _t
    _t.triton_hash_with_backend = functools.cache(_stable_backend_hash)
    notes.append("patched torch.utils._triton.triton_hash_with_backend "
                 "(no GPU target to query)")

    _installed = True
    return notes
