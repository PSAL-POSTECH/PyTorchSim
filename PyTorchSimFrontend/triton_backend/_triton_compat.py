"""Make torch 2.8's Inductor codegen work against tnpu's triton 3.6, CPU-only.

TWO SKEWS, BOTH STRUCTURAL
--------------------------
1. VERSION. Inductor in torch 2.8 targets the triton ~3.3 API. tnpu pins triton
   3.6 and that pin is not negotiable: 3.6 is what pins LLVM 23, and both sides
   of the textual IR seam must be the same LLVM (triton-npu/setup/versions.env).
   So the frontend has to bend, not the backend.

2. NO GPU. `triton_hash_with_backend()` asks the triton runtime driver for the
   *current target*, which on a machine with no GPU has nothing to answer with.
   We never launch through triton's runtime -- the kernel is compiled ahead of
   time to a RISC-V ELF -- so the value is only a cache-key ingredient.

Both are handled by replacing `torch.utils._triton.triton_hash_with_backend`
with a deterministic string. It is a monkeypatch, and it is the cheap half of a
real choice: the alternative is installing a torch-2.8-compatible triton (3.3.x)
in the driver interpreter purely for codegen, keeping 3.6 in the tnpu venv for
stage 1. That works because the two interpreters exchange only SOURCE TEXT and
never share objects -- but it means two tritons to keep straight, so it is worth
doing only if the API drift turns out to be wider than this one symbol.
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


def _needs_hash_patch():
    """True when triton_hash_with_backend cannot work here."""
    try:
        mod = importlib.import_module("triton.compiler.compiler")
    except Exception:  # noqa: BLE001
        return True
    return not hasattr(mod, "triton_key")


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

    if _needs_hash_patch():
        # `triton_key` is imported from triton.compiler.compiler by SEVERAL torch
        # call sites (codecache.CacheBase.get_system, _triton.triton_hash_with_
        # backend, ...), each with its own local import. Supplying the symbol on
        # the triton side satisfies all of them at once instead of chasing every
        # call site; it is a cache-key ingredient, so any stable string will do.
        mod = importlib.import_module("triton.compiler.compiler")
        mod.triton_key = _stable_backend_hash
        notes.append("injected triton.compiler.compiler.triton_key "
                     "(removed in triton 3.6; torch 2.8 still imports it)")

    # Separately: triton_hash_with_backend also asks the triton runtime driver
    # for the current target, which needs a GPU. We compile ahead of time to a
    # RISC-V ELF and never use triton's runtime, so short-circuit it.
    import torch.utils._triton as _t
    _t.triton_hash_with_backend = functools.cache(_stable_backend_hash)
    notes.append("patched torch.utils._triton.triton_hash_with_backend "
                 "(no GPU target to query)")

    _installed = True
    return notes
