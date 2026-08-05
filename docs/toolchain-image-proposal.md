# Proposal: build the tnpu toolchain in the image, not into it

Status: proposal. Nothing here is implemented.

## The claim

`torchsim_tnpu_base` is a Docker image whose build step downloads a 1.8 GiB
build cache from a private GitHub release and unpacks it to absolute paths. A
Docker image layer *is* a build cache. We are running a second, hand-maintained
cache inside the first one, and every problem in this area comes from that.

## What exists today

```
PSAL-POSTECH/triton-npu
  setup/versions.env        8 pins, maintained by hand
  setup/restore.sh          ~300 lines of bash
  release toolchain-llvm23  llvm23-install.tar.gz     1.18 GiB
                            triton-runtime.tar.gz     0.34 GiB
                            spike-install.tar.gz      0.01 GiB
                            MANIFEST.txt              what the assets were built from,
                                                      written by hand
PyTorchSim
  Dockerfile.tnpu           runs restore.sh --prebuilt inside the image build
  thirdparty/triton-npu.json  pins all of the above as one commit sha
```

The assets are produced by `setup/package.sh` on a developer machine and
uploaded to the release by hand.

## Four costs, all observed

**1. The cache is maintained by a human.** `versions.env` carries
`TRITON_SHARED_PREBUILT_SHA`, whose comment records that it "had gone stale
twice over" — it is a note about what the uploaded tarball contains, and
nothing checks it. `MANIFEST.txt` is the same shape of problem.

**2. Absolute paths make unpacking a restore, not a copy.** The triton install
is editable, so `__editable__*_finder.py` holds `/workspace/triton-src/python/
triton` and the venv shebangs hold `/workspace/mlir-env/bin/python`. A tarball
must land exactly where it was built. Renaming one directory (2026-08-03,
`/workspace/triton` → `/workspace/triton-src`, because `triton` shadowed the
package for anything run from `/workspace`) required patching the editable
finder, re-linking the backend symlinks, repackaging the 340 MB tarball and
re-uploading it, and keeping the old asset under a `.flat-triton` name.

**3. `--prebuilt` turns off more than it needs to.** It sets `STEPS=(layout)`,
which skips both `tritonenv` (the real build, correctly skipped) and
`tritonshared` (a git clone: 1 second, 6 MB). To compensate, `restore.sh` grew
a special case that fetches three files — `backend/{compiler.py,driver.py,
name.conf}` — from a *different repository at a different commit*
(`facebookincubator/triton-shared` at `TRITON_SHARED_BACKEND_SHA`), because
`raw.githubusercontent` cannot serve our private fork. Twelve lines of comment
defend the assumption that "every fork commit touches only include/ and lib/".
On 2026-08-04 a fork commit touched `backend/compiler.py`; the assumption broke
and the prebuilt path silently kept using upstream's older file.

**4. Pins multiply.** Three variables name the same repository —
`TRITON_SHARED_SHA`, `TRITON_SHARED_BACKEND_SHA`, `TRITON_SHARED_PREBUILT_SHA`
— and only the first is authoritative.

## The proposal

Build the toolchain in `Dockerfile.tnpu` from source, and let the image layer be
the cache.

```dockerfile
ARG LLVM_SHA=...
ARG TRITON_SHA=df38505...
ARG TRITON_SHARED_SHA=...
ARG SPIKE_SHA=...

RUN git clone --depth 1 ... && cmake ... && ninja install     # llvm 23
RUN TRITON_PLUGIN_DIRS=/src/triton_shared pip install -e /src/triton
RUN ... spike, riscv-pk
```

What disappears:

| | |
|---|---|
| `setup/restore.sh` | ~300 lines |
| release assets + `package.sh` + manual upload | 1.8 GiB, and the step that produces it |
| `MANIFEST.txt`, `TRITON_SHARED_PREBUILT_SHA` | records of what a tarball holds |
| `TRITON_SHARED_BACKEND_SHA` + the three-file fetch | the special case in cost 3 |
| "restore to absolute paths" | the paths are only ever inside the image |
| `GITHUB_TOKEN` for release assets | still needed for the private fork clone |

What it costs: the first build of a given pin set compiles LLVM 23. Every later
build with the same pins is a registry pull. `ensure-tnpu-base` already skips
the build when the tag exists (`docker manifest inspect`), so the trigger is
unchanged — only the miss is more expensive.

## The open question: walltime

Jobs run on the PSAL Slurm farm, where `big` is 16 cores with a 2 hour
`--time`. An LLVM 23 build with MLIR will not finish in that. Options, in the
order I would try them:

1. **A dedicated bucket** with a longer `--time` for this one job. Config-only
   change on the runner farm (`~/.ghr/config.toml`); nothing in this repo moves.
2. **Split the image**: one layer per component (`llvm23` / `triton` / `spike`),
   each its own tag and its own job. Each fits in 2 hours, and a pin move
   rebuilds only its own layer instead of all of them.
3. **Keep LLVM prebuilt, build the rest.** LLVM is the only genuinely long one
   and its pin almost never moves; triton + triton_shared + spike are minutes.
   This keeps one tarball and deletes the other two, plus the whole special case
   in cost 3.

Option 3 is the smallest step that removes most of the pain, and is worth
measuring before committing to 1 or 2.

## What this proposal is not

It does not change what the toolchain *is* — same LLVM, same triton pin, same
passes. It changes only how the environment is assembled, and it should be
judged on whether the four costs above are worth ~300 lines of bash and a manual
upload step.

## Sequencing

This should not block the libdevice fix. That fix needs the three-file special
case gone, which is a ~15 line change to `restore.sh` (re-enable the
`tritonshared` step, delete the fetch block). Doing that first makes the
proposal smaller, not larger: it removes cost 3 on its own.
