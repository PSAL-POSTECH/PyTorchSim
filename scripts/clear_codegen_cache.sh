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

echo "Clearing $DUMP_PATH/.torchinductor and per-source-hash dirs"
rm -rf "$DUMP_PATH/.torchinductor"

# Per-source-hash dirs are an 11-char alphanumeric prefix
# (extension_codecache.hash_prefix). Match by length+charset so we don't
# touch anything else a developer may have parked under outputs/.
find "$DUMP_PATH" -mindepth 1 -maxdepth 1 -type d \
    -regextype posix-egrep -regex '.*/[a-z0-9]{11}$' \
    -exec rm -rf {} +

echo "Done."
