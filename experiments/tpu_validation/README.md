# tpu_validation

Runs the shapes of a real-TPU table through PyTorchSim and reports how the simulated latency
compares with the measured one. One command per op; gemm and softmax go through the same code and
get the same numbers, so their results can be read side by side.

    TPU table (tpu-perf-archive) ──> run.py ──> sim_op.py per shape ──> results.csv ──> compare.py ──> compare.csv + summary

## Setup

1. A working PyTorchSim environment: the one in which `python $TORCHSIM_DIR/experiments/gemm.py`
   runs. `TORCHSIM_DIR` must be set; `TORCHSIM_COMPILE_DIR` is recorded if set.
2. Read access to `PSAL-POSTECH/tpu-perf-archive` over SSH. The first run clones it next to these
   scripts (`tpu-perf-archive/`, shallow) and later runs reuse that clone; `git -C tpu-perf-archive pull`
   picks up new tables. To use a clone you already have, `export TPU_PERF_ARCHIVE=/path/to/it`.

Nothing else to install; the scripts use only the standard library.

## Run

    python run.py gemm    --tag g1 --keep KxN=768x768,2048x3072
    python run.py softmax --tag s1 --keep C=128,512,2048,8192 --drop R=1
    python compare.py g1 s1                  # both summaries, then one line each side by side

`run.py` simulates every shape the filters leave, `--jobs` at a time (default 8), and prints the
comparison when it finishes. Running the same command again continues where it stopped: finished
shapes are skipped, failed ones retried. Without `--keep` it runs the whole table (gemm 230 shapes,
softmax 157), which takes many hours; the large shapes dominate.

| option | meaning |
|---|---|
| `--tag T` | names the run; everything lands in `runs/T/` |
| `--keep COLS=VALUES` | only these shapes. `C=128,512`, or several columns at once: `KxN=768x768,2048x3072`. Repeatable; a shape must match every `--keep` |
| `--drop COLS=VALUES` | skip these shapes, same syntax |
| `--set KEY=VALUE` | change one line of the config, e.g. `--set vpu_num_lanes=128 --set core_freq_mhz=1750`. Repeatable |
| `--config FILE` | base config, default `configs/tpuv6e_nocache.yml` |
| `--timeout S` | seconds per shape, default 7200; a shape over it is killed and logged in `failed.csv` |
| `--out DIR` | where the tag directories live, default `runs/` here. Keep it on a real disk, not a RAM tmpfs |

A tag is one op on one config. Reusing a tag with a different op or config is refused, so two
settings never mix in one `results.csv`; use a new tag for each variant.

## What a run leaves

    runs/<tag>/
      config.yml     the exact config every shape ran on ($TOGSIM_DIR expanded, --set applied)
      meta.json      op, TPU table, frontend / compiler / archive commits, every command line
      results.csv    shape, cycles, sim_ns, dram_reads, dram_writes, sa_util_pct, wall_s
      failed.csv     shape, rc (or "timeout"), last log line
      jobs/<shape>/  stdout.log.gz (stdout.log if the shape failed)
      compare.csv    shape, tpu_ns, sim_ns, sim_cycles, sim_over_tpu, tpu_status + TPU columns

## How the numbers are defined

- `tpu_ns` is the table's `latency_ns`: device time of one execution, the median over the traced
  executions (tpu-perf-archive README).
- `sim_ns = cycles * 1000 / core_freq_mhz`, with `core_freq_mhz` read from the run's `config.yml`.
- `sim_over_tpu = sim_ns / tpu_ns`. Below 1 the simulator is faster than the TPU.
- The summary gives median, IQR, min, max and MAPE (mean of `|sim_over_tpu - 1|`) for all simulated
  shapes, for those whose TPU row has `status == ok`, and for the op's buckets (gemm: M range and
  memory/compute bound; softmax: element count and C).
- The simulated program is the one the table measured, with fp16 in place of bf16:
  gemm `torch.matmul(a, b)`, softmax `torch.softmax(x, -1, dtype=float32).to(float16)`.

Two summaries are comparable only if they ran on the same `config.yml` and the same commits;
`compare.py` prints the commits and any `--set` next to each tag.

## Known issue

Softmax shapes with `R = 1` hang in gem5 while the kernel is being compiled (seen on frontend 3984351 with
compiler 670ae21, on both lane counts). Pass `--drop R=1`, or they each hold a job slot until
`--timeout`.

## Adding an op

Add its entry to `OPS` in `ops.py` (shape columns, work size, summary buckets) and its program to
`sim_op.py`. The table must exist as `data/v6e/<op>/torchxla.csv` in the archive.
