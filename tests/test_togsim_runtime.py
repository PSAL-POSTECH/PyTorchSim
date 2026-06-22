"""P3 task 5: the TOGSim C6 runtime + loader (togsim_runtime.cc / togsim_loader.h).

Builds a producer `.so` from a post-vcix fixture, links the real C6 runtime, runs
the loader (`run_producer`) against the `.so`, and checks the recorded trace:
DRAM addresses are resolved (base[arg_id] + offset*elem_bytes), compute cycles
are looked up from the cycle table, and every wait gets a handle a dma minted.

Skipped unless the MLIR bindings, `mlir-translate`, a C++ compiler, and a
post-vcix `.mlir` fixture (`TOGSIM_SKELETON_FIXTURE`) are available.
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
_RUNTIME = _ROOT / "TOGSim" / "src" / "togsim_runtime.cc"


def _mlir_translate():
    return os.path.join(os.environ.get("TORCHSIM_LLVM_PATH", "/usr/bin"),
                        "mlir-translate")


def _tools_ready():
    return (importlib.util.find_spec("mlir") is not None
            and os.path.isfile(_mlir_translate())
            and shutil.which(_CXX) is not None
            and _RUNTIME.is_file())


def _fixture():
    fix = os.environ.get("TOGSIM_SKELETON_FIXTURE")
    if not fix or not os.path.isfile(fix):
        pytest.skip("set TOGSIM_SKELETON_FIXTURE to a post-vcix kernel .mlir")
    return fix


# Drives the loader with known tensor bases + a synthetic cycle table, then
# checks the recorded trace. Tailored to a single-output-tile GEMM (256^3):
# 3 dmas A/B/C at offset 0 -> addr == base; args 0/1/2; dirs load/load/store.
_MAIN = r'''
#include <cstdio>
#include <cstdint>
#include <utility>
#include <vector>
#include "togsim_loader.h"
using namespace togsim;
int main(int argc, char** argv) {
  uint64_t bases[3] = {0x1000, 0x2000, 0x3000};
  int64_t  cyc[3]   = {100, 200, 300};
  int64_t  ovl[3]   = {0, 200, 172};
  RunResult r = run_producer(argv[1], nullptr, 0, bases, 3, cyc, ovl, 3, 1);
  if (!r.ok) { printf("run failed\n"); return 2; }
  int ndisp=0, nd=0, nc=0, nm=0, fail=0;
  std::vector<uint64_t> dma_a; std::vector<int> dma_arg, dma_dir;
  std::vector<std::pair<int,uint64_t>> async_tags;  // (tag_id, tag_slot) of async dmas
  for (auto& t : r.trace) {
    if (t.kind == TraceRec::TILE_BEGIN) ndisp++;   // one per work-item
    else if (t.kind == TraceRec::DMA) {
      nd++; dma_a.push_back(t.addr);
      dma_arg.push_back(t.arg_id); dma_dir.push_back(t.dir);
      if (t.is_async) async_tags.push_back({t.tag_id, t.tag_slot});
    } else if (t.kind == TraceRec::COMPUTE) {
      nc++;
      int64_t want = (t.tile_id < 3) ? cyc[t.tile_id] : -1;
      if (t.cycle != want) { printf("compute %lu cyc %ld!=%ld\n",
          (unsigned long)t.tile_id, (long)t.cycle, (long)want); fail++; }
    } else if (t.kind == TraceRec::MEMORY_BAR) {
      nm++; bool ok=false;
      for (auto& k : async_tags) if (k.first==t.tag_id && k.second==t.tag_slot) ok=true;
      if (!ok) { printf("membar tag (%d,%lu) pairs no async dma\n",
          t.tag_id, (unsigned long)t.tag_slot); fail++; }
    }
  }
  const uint64_t exp[3] = {0x1000, 0x2000, 0x3000};
  const int ea[3] = {0,1,2}, ed[3] = {0,0,1};
  for (int i = 0; i < nd && i < 3; ++i)
    if (dma_a[i]!=exp[i] || dma_arg[i]!=ea[i] || dma_dir[i]!=ed[i]) {
      printf("dma[%d] addr=%#lx arg=%d dir=%d\n", i,
             (unsigned long)dma_a[i], dma_arg[i], dma_dir[i]); fail++;
    }
  printf("dispatch=%d dma=%d compute=%d membar=%d fail=%d\n", ndisp, nd, nc, nm, fail);
  printf(fail ? "RESULT FAIL\n" : "RESULT PASS\n");
  return fail ? 1 : 0;
}
'''


@pytest.mark.skipif(not _tools_ready(),
                    reason="need mlir bindings + mlir-translate + C++ compiler + runtime")
def test_runtime_loads_and_records():
    fix = _fixture()
    sys.path.insert(0, str(_ROOT))
    from PyTorchSimFrontend.mlir.passes import lower_to_emitc as c4

    with tempfile.TemporaryDirectory() as d:
        so = os.path.join(d, "trace.so")
        c4.build_trace_so(fix, so)

        main_cpp = os.path.join(d, "main.cpp")
        binp = os.path.join(d, "runtime_test")
        with open(main_cpp, "w") as fh:
            fh.write(_MAIN)
        build = subprocess.run(
            [_CXX, "-std=gnu++17", "-O2", "-rdynamic", "-I", str(_INCLUDE),
             main_cpp, str(_RUNTIME), "-o", binp, "-ldl"],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr

        run = subprocess.run([binp, so], capture_output=True, text=True)
        out = run.stdout
        assert "RESULT PASS" in out, out + run.stderr
        assert run.returncode == 0, out
        # at least the GEMM's 3 dmas were recorded with resolved addresses.
        line = [l for l in out.splitlines() if l.startswith("dispatch=")][0]
        counts = dict(kv.split("=") for kv in line.split())
        assert int(counts["dma"]) >= 1
        assert int(counts["compute"]) >= 1
        assert int(counts["fail"]) == 0


_SIM_MAIN = r'''
#include <cstdio>
#include <cstdint>
#include "togsim_loader.h"
using namespace togsim;
int main(int argc, char** argv) {
  uint64_t bases[3] = {0x1000, 0x2000, 0x3000};
  int64_t  cyc[3]   = {100, 200, 300};
  int64_t  ovl[3]   = {0, 200, 172};
  RunResult r = run_producer(argv[1], nullptr, 0, bases, 3, cyc, ovl, 3, 1);
  if (!r.ok) { printf("run failed\n"); return 2; }
  TimingParams p; p.dma_latency = 100;
  SimResult s = simulate(r, p);
  // serial baseline: no overlap at all.
  uint64_t serial = 0;
  for (auto& t : r.trace) {
    if (t.kind == TraceRec::DMA) serial += p.dma_latency;
    else if (t.kind == TraceRec::COMPUTE) serial += (uint64_t)t.cycle;
  }
  printf("SIM total=%lu compute=%d dma=%d serial=%lu\n",
         (unsigned long)s.total_cycle, s.n_compute, s.n_dma, (unsigned long)serial);
  // The trace is schedulable into cycles; overlap (dma||compute, compute
  // pipelining) makes it no worse than the fully-serial baseline.
  bool ok = s.total_cycle > 0 && s.n_compute > 0 && s.total_cycle <= serial;
  printf(ok ? "RESULT PASS\n" : "RESULT FAIL\n");
  return ok ? 0 : 1;
}
'''


@pytest.mark.skipif(not _tools_ready(),
                    reason="need mlir bindings + mlir-translate + C++ compiler + runtime")
def test_simulate_produces_cycles():
    fix = _fixture()
    sys.path.insert(0, str(_ROOT))
    from PyTorchSimFrontend.mlir.passes import lower_to_emitc as c4

    with tempfile.TemporaryDirectory() as d:
        so = os.path.join(d, "trace.so")
        c4.build_trace_so(fix, so)
        main_cpp = os.path.join(d, "sim.cpp")
        binp = os.path.join(d, "sim_test")
        with open(main_cpp, "w") as fh:
            fh.write(_SIM_MAIN)
        build = subprocess.run(
            [_CXX, "-std=gnu++17", "-O2", "-rdynamic", "-I", str(_INCLUDE),
             main_cpp, str(_RUNTIME), "-o", binp, "-ldl"],
            capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        run = subprocess.run([binp, so], capture_output=True, text=True)
        assert "RESULT PASS" in run.stdout, run.stdout + run.stderr
        assert run.returncode == 0, run.stdout
