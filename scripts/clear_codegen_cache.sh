#!/usr/bin/env bash
# Clear PyTorchSim's codegen caches so the next torch.compile run regenerates
# the wrapper Python and the per-kernel MLIR. Run this whenever you edit
# anything that affects emitted MLIR (PyTorchSimFrontend/mlir/*, lowering
# rules, codegen backend, etc.) -- otherwise the previous compile is replayed
# byte-for-byte from $TORCHSIM_DUMP_PATH and your change appears not to take.
#
# Wipes:
#   $TORCHSIM_DUMP_PATH/.torchinductor      (Inductor compile cache, points
#                                            here via TORCHINDUCTOR_CACHE_DIR
#                                            set in extension_config.py)
#   $TORCHSIM_DUMP_PATH/<11-char-hash>/     (per-source MLIR/wrapper dirs,
#                                            keyed by hash_prefix(src) in
#                                            extension_codecache.py)
#   $TORCHSIM_DUMP_PATH/triton_<hash>/      (the Triton route's per-kernel
#                                            artifacts: spec, staged IR, ELF)
#
# WHY THE TRITON ONES MATTER MORE THAN THEY LOOK. That hash is of the INDUCTOR
# SOURCE, so a fix anywhere BELOW it -- a tnpu pass, triton-shared -- leaves the
# hash alone and the launcher reuses the ELF it already has. Measured: two runs
# of test_transformer.py reported the same divergence while the kernel, given
# the model's own recorded inputs, passed standalone at 2.7e-07; the artifacts
# were a day old. A compiler that moved and a cache that did not is a wrong
# measurement, not a slow one.
#
# Does NOT touch:
#   $TORCHSIM_LOG_PATH (togsim_results/, just simulation logs)
#   Anything outside $TORCHSIM_DUMP_PATH
#
# Usage:
#   scripts/clear_codegen_cache.sh
set -euo pipefail

DUMP_PATH="${TORCHSIM_DUMP_PATH:-${TORCHSIM_DIR:-/workspace/PyTorchSim}/outputs}"

if [[ ! -d "$DUMP_PATH" ]]; then
    echo "No cache at $DUMP_PATH; nothing to clear."
    exit 0
fi

echo "Clearing $DUMP_PATH/.torchinductor, per-source-hash and triton_* dirs"
rm -rf "$DUMP_PATH/.torchinductor"

# Per-source-hash dirs are an 11-char alphanumeric prefix
# (extension_codecache.hash_prefix). Match by length+charset so we don't
# touch anything else a developer may have parked under outputs/.
find "$DUMP_PATH" -mindepth 1 -maxdepth 1 -type d \
    -regextype posix-egrep -regex '.*/[a-z0-9]{11}$' \
    -exec rm -rf {} +

# The Triton route's dirs, same reasoning and a different prefix.
find "$DUMP_PATH" -mindepth 1 -maxdepth 1 -type d \
    -regextype posix-egrep -regex '.*/triton_[a-z0-9]+$' \
    -exec rm -rf {} +

echo "Done."
