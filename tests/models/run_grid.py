"""Run the real-size model tests at the grid points the TPU v6e table measures, one process per point.

LLM rows run one decoder layer (``--part layer``): a prefill of S tokens (``--seq S``) and, where the model
takes a KV cache, one decode step after a cache of S (``--decode S``), S in 128 .. 4096 (the simulator's cap)
and within the model's position table. Vision and fixed-input rows run once at their test's own input.
Points run one at a time; each writes its log under ``--out`` and one line to ``--out``/summary.tsv:
name, mode, S, rc, passed, failed, wall seconds, and the summed ``simulated: N cycles`` of its log.

    python tests/models/run_grid.py --out /opt/torchsim-local/run/<dir> --points rep [--only gpt2,qwen3_8b]
    python tests/models/run_grid.py --out /opt/torchsim-local/run/<dir> --points all --dry-run

A point already in summary.tsv is skipped, so a stopped sweep resumes where it was; ``--rerun`` runs it again.
Run it from the environment the tests need (TORCHSIM_DIR at this checkout, the compiler, the simulators); it
runs this checkout's tests, starts its own children and touches no other process.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time

GRID = (128, 256, 512, 1024, 2048, 4096)
REP = 1024
LLM = ("seq", "decode")

#: name (the TPU table's model id) -> (test file under tests/models, its arguments, modes, largest S).
#: modes () is a fixed-input row; largest S is the position table where it is below the grid's cap.
TESTS = {
    "bert": ("BERT/test_bert.py", ["--part", "layer"], ("seq",), 512),
    "gpt2": ("GPT2/test_gpt2.py", ["--part", "layer"], LLM, 1024),
    "gemma_7b": ("Gemma/test_gemma.py", ["--preset", "7b", "--part", "layer"], LLM, None),
    "gemma2_2b": ("Gemma/test_gemma.py", ["--preset", "2-2b", "--part", "layer"], LLM, None),
    "gemma3_1b": ("Gemma/test_gemma.py", ["--preset", "3-1b", "--part", "layer"], LLM, None),
    "moonlight": ("Kimi/test_kimi.py", ["--rung", "moonlight", "--part", "layer"], LLM, None),
    "deepseek_v3": ("DeepSeek/test_deepseek_v3_base.py", ["--part", "layer"], LLM, None),
    "llama2_7b": ("Llama/test_llama.py", ["--part", "layer"], LLM, None),
    "llama3_8b": ("Llama/test_llama3.py", ["--part", "layer"], LLM, None),
    "llama31_8b": ("Llama/test_llama3x.py", ["--preset", "3.1-8b", "--part", "layer"], LLM, None),
    "llama4_scout": ("Llama/test_llama4.py", ["--part", "layer"], LLM, None),
    "mixtral_8x7b": ("Mixtral8x7B/test_mistral.py", ["--part", "layer"], LLM, None),
    "qwen3_8b": ("Qwen/test_qwen.py", ["--preset", "3-8b", "--part", "layer"], LLM, None),
    "qwen15_moe": ("Qwen/test_qwen.py", ["--preset", "2-moe", "--part", "layer"], LLM, None),
    "qwen3_vl_moe": ("Qwen/test_qwen_next.py", ["--family", "qwen3_vl_moe", "--part", "layer"], LLM, None),
    "qwen35_9b": ("Qwen/test_qwen_next.py", ["--family", "qwen3_5", "--part", "layer"], LLM, None),
    "mllama_11b_vision": ("Llama/test_mllama.py", ["--part", "vision"], (), None),
    "mllama_11b_text": ("Llama/test_mllama.py", ["--part", "text"], (), None),
    "qwen2_vl_2b": ("Qwen/test_qwen_vl.py", ["--version", "2"], (), None),
    "qwen25_vl_3b": ("Qwen/test_qwen_vl.py", ["--version", "2.5"], (), None),
    "sd15_unet": ("Diffusion/test_diffusion.py", [], (), None),
    "yolov5n": ("Yolo/test_yolo.py", ["--version", "v5"], (), None),
    "yolov8n": ("Yolo/test_yolo.py", ["--version", "v8"], (), None),
    "yolov5s": ("Yolov5/test_yolov5.py", [], (), None),
    "clip_b32": ("test_clip.py", [], (), None),
    "convnextv2_tiny": ("test_convnextv2.py", [], (), None),
    "swinv2_tiny": ("test_swinv2.py", [], (), None),
}

COLUMNS = ("name", "mode", "S", "rc", "passed", "failed", "wall_s", "cycles")
CYCLES = re.compile(r"simulated: (\d+) cycles")


def points(names, which):
    """(name, mode, S) of every point of `names`: "rep" is prefill and decode at REP (or the largest S the
    model takes below it) plus every fixed row, "all" is the representative points first, then the rest."""
    rep, rest = [], []
    for name in names:
        _, _, modes, top = TESTS[name]
        if not modes:
            rep.append((name, "fixed", 0))
            continue
        sizes = [s for s in GRID if top is None or s <= top]
        r = max(s for s in sizes if s <= REP)
        for mode in modes:
            rep.append((name, mode, r))
            rest += [(name, mode, s) for s in sizes if s != r]
    return rep if which == "rep" else rep + rest


def command(root, name, mode, s):
    """The argv that runs one point."""
    path, args, _, _ = TESTS[name]
    grid = [] if mode == "fixed" else [f"--{mode}", str(s)]
    return [sys.executable, os.path.join(root, "tests", "models", path), *args, *grid]


def tag(name, mode, s):
    return f"{name}.{mode}" if mode == "fixed" else f"{name}.{mode}{s}"


def done(summary):
    """The (name, mode, S) already written to `summary`."""
    if not os.path.exists(summary):
        return set()
    with open(summary) as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    return {(r[0], r[1], int(r[2])) for r in rows[1:]}


def run_point(root, out, name, mode, s, timeout):
    """Run one point in its own session; return its summary row. A timeout ends that session only."""
    log = os.path.join(out, "logs", tag(name, mode, s) + ".log")
    env = dict(os.environ, TORCHSIM_LOG_PATH=os.path.join(out, "togsim", tag(name, mode, s)))
    os.makedirs(env["TORCHSIM_LOG_PATH"], exist_ok=True)
    start = time.time()
    with open(log, "w") as f:
        proc = subprocess.Popen(command(root, name, mode, s), cwd=root, env=env, stdout=f,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout or None)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            rc = proc.wait()
    wall = time.time() - start
    with open(log, errors="replace") as f:
        text = f.read()
    passed = text.count("Test Passed")
    failed = text.count("Test Failed") + text.count("Traceback")
    cycles = sum(int(n) for n in CYCLES.findall(text))
    return (name, mode, s, rc, passed, failed, f"{wall:.0f}", cycles)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="output directory, under /opt/torchsim-local/run/")
    parser.add_argument("--points", choices=["rep", "all"], default="rep")
    parser.add_argument("--only", default=None, help=f"comma-separated names out of: {', '.join(TESTS)}")
    parser.add_argument("--timeout", type=int, default=0, help="seconds per point, 0 for none")
    parser.add_argument("--rerun", action="store_true", help="run points summary.tsv already holds")
    parser.add_argument("--dry-run", action="store_true", help="print the command of every point and stop")
    args = parser.parse_args()

    out = os.path.realpath(args.out)
    if out == "/tmp" or out.startswith("/tmp/"):
        parser.error("--out is a RAM tmpfs under /tmp; put it under /opt/torchsim-local/run/")
    names = args.only.split(",") if args.only else list(TESTS)
    unknown = [n for n in names if n not in TESTS]
    if unknown:
        parser.error(f"unknown --only {unknown}; known: {', '.join(TESTS)}")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    summary = os.path.join(out, "summary.tsv")
    skip = set() if args.rerun else done(summary)
    todo = [p for p in points(names, args.points) if p not in skip]
    if args.dry_run:
        for name, mode, s in todo:
            log = os.path.join(out, "logs", tag(name, mode, s) + ".log")
            print(f"cd {root} && TORCHSIM_LOG_PATH={os.path.join(out, 'togsim', tag(name, mode, s))} "
                  f"{' '.join(command(root, name, mode, s))} > {log} 2>&1")
        print(f"# {len(todo)} points")
        return

    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    if not os.path.exists(summary):
        with open(summary, "w") as f:
            f.write("\t".join(COLUMNS) + "\n")
    for i, (name, mode, s) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {tag(name, mode, s)} ...", flush=True)
        row = run_point(root, out, name, mode, s, args.timeout)
        with open(summary, "a") as f:
            f.write("\t".join(str(v) for v in row) + "\n")
        print("  " + "  ".join(f"{k}={v}" for k, v in zip(COLUMNS, row)), flush=True)


if __name__ == "__main__":
    main()
