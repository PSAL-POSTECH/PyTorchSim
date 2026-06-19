"""Tests for the C4 emitc lowering + compiled .so trace producer (P2).

The pipeline under test (docs/design/togsim_cpp_trace.md, sec 5-7):

    post-vcix .mlir --build_skeleton--> skeleton+API
                    --lower_to_emitc--> EmitC module
                    --mlir-translate--> C++
                    --g++ -shared----> trace .so  (exports togsim_kernel;
                                                    togsim_* left undefined)

`test_build_trace_so` builds the .so and checks the EmitC/symbol-table shape.
`test_trace_so_runs` additionally dlopens it against a stub runtime and confirms
the producer executes and emits a non-empty deterministic trace.

Both are skipped unless the MLIR bindings, `mlir-translate` (from
TORCHSIM_LLVM_PATH), a host C++ compiler, AND a post-vcix `.mlir` fixture (via
`TOGSIM_SKELETON_FIXTURE`) are available -- the same fixture used by
test_togsim_skeleton.py.
"""
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CXX = os.environ.get("CXX", "g++")
_INCLUDE = _ROOT / "TOGSim" / "include"


def _mlir_translate():
    return os.path.join(os.environ.get("TORCHSIM_LLVM_PATH", "/usr/bin"),
                        "mlir-translate")


def _tools_ready():
    return (importlib.util.find_spec("mlir") is not None
            and os.path.isfile(_mlir_translate())
            and shutil.which(_CXX) is not None)


def _fixture():
    fix = os.environ.get("TOGSIM_SKELETON_FIXTURE")
    if not fix or not os.path.isfile(fix):
        pytest.skip("set TOGSIM_SKELETON_FIXTURE to a post-vcix kernel .mlir")
    return fix


_HARNESS = r'''
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include "togsim_runtime.h"
static int n_dma=0, n_membar=0, n_compute=0, n_core=0, bad=0;
extern "C" {
void togsim_dma(EmitCtx*, int32_t, int32_t, uint64_t, int32_t,
                const int64_t*, const int64_t*, int32_t, int32_t,
                int32_t, uint64_t, const int64_t*, int32_t,
                const int64_t*, int32_t){ ++n_dma; }
void togsim_compute(EmitCtx*, uint64_t, int32_t, int32_t, const int64_t*,
                    const int64_t*, int32_t, const int64_t*, int32_t){ ++n_compute; }
void togsim_memory_barrier(EmitCtx*, int32_t tag_id, uint64_t, const int64_t*, int32_t){
  ++n_membar; if(tag_id<0) ++bad; }   // tag_id pairs it with its async dma
int32_t togsim_core_alloc(EmitCtx*){ return n_core++; }   // count + assign a core
void togsim_compute_barrier(EmitCtx*){}
}
int main(int argc, char** argv){
  void* h = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
  if(!h){ printf("dlopen failed: %s\n", dlerror()); return 2; }
  auto emit = (void(*)(EmitCtx*, int64_t*, int32_t))dlsym(h, "togsim_kernel");
  if(!emit){ printf("dlsym failed: %s\n", dlerror()); return 3; }
  emit(nullptr, nullptr, 0);
  printf("TRACE core=%d dma=%d membar=%d compute=%d bad=%d\n",
         n_core, n_dma, n_membar, n_compute, bad);
  return 0;
}
'''


@pytest.mark.skipif(not _tools_ready(),
                    reason="need mlir bindings + mlir-translate + C++ compiler")
def test_build_trace_so():
    fix = _fixture()
    sys.path.insert(0, str(_ROOT))
    from PyTorchSimFrontend.mlir.passes import lower_to_emitc as c4

    with tempfile.TemporaryDirectory() as d:
        so = os.path.join(d, "trace.so")
        emitc_text = c4.build_trace_so(fix, so)
        assert os.path.isfile(so)

        # EmitC form: one entry func, dma/memory_barrier/compute as call_opaque targets.
        assert "emitc.func" in emitc_text
        assert ("@%s" % c4.ENTRY) in emitc_text
        assert 'emitc.call_opaque "togsim_dma"' in emitc_text
        assert 'emitc.call_opaque "togsim_memory_barrier"' in emitc_text
        assert 'emitc.call_opaque "togsim_compute"' in emitc_text

        # Symbol table: entry exported (defined, text), runtime hooks undefined
        # so the TOGSim loader resolves them at dlopen.
        nm = subprocess.run(["nm", "-D", so], capture_output=True, text=True).stdout
        syms = {parts[-1]: parts[-2] for parts in
                (ln.split() for ln in nm.splitlines()) if len(parts) >= 2}
        assert syms.get("togsim_kernel") == "T", nm
        assert syms.get("togsim_dma") == "U", nm
        assert syms.get("togsim_core_alloc") == "U", nm
        assert syms.get("togsim_memory_barrier") == "U", nm
        # The per-work-item core alloc is emitted.
        assert 'emitc.call_opaque "togsim_core_alloc"' in emitc_text


@pytest.mark.skipif(not _tools_ready(),
                    reason="need mlir bindings + mlir-translate + C++ compiler")
def test_trace_so_runs():
    fix = _fixture()
    sys.path.insert(0, str(_ROOT))
    from PyTorchSimFrontend.mlir.passes import lower_to_emitc as c4

    with tempfile.TemporaryDirectory() as d:
        so = os.path.join(d, "trace.so")
        c4.build_trace_so(fix, so)

        harness_cpp = os.path.join(d, "harness.cpp")
        harness_bin = os.path.join(d, "harness")
        with open(harness_cpp, "w") as fh:
            fh.write(_HARNESS)
        # -rdynamic so the harness's togsim_* are visible to the dlopened .so.
        build = subprocess.run(
            [_CXX, "-std=gnu++17", "-O2", "-rdynamic", "-I", str(_INCLUDE),
             harness_cpp, "-o", harness_bin, "-ldl"],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr

        run = subprocess.run([harness_bin, so], capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        out = run.stdout.strip()
        assert out.startswith("TRACE "), out
        counts = dict(kv.split("=") for kv in out.split()[1:])
        # The producer ran and emitted a real trace, with >=1 work-item (core alloc).
        assert int(counts["core"]) >= 1
        assert int(counts["dma"]) >= 1
        assert int(counts["compute"]) >= 1
        # Async loads are synced by explicit memory barriers, each carrying a
        # valid (non-negative) tag_id that pairs it with its dma.
        assert int(counts["membar"]) >= 1, out
        assert int(counts["bad"]) == 0, out
