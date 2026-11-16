"""use_machine(): TORCHSIM_TARGET as tpuv6e with a smaller scratchpad, written to a temp machine file, for the tests
that need the matrix unit (tpuv3_32x32 has none) and a scratchpad small enough that a candidate tile overflows."""
import os
import tempfile

import yaml


def use_machine(budget_kb, base="tpuv6e"):
    """Point TORCHSIM_TARGET at `base` with budget_kb of scratchpad a lane left for a kernel; call before compiling."""
    os.environ["TORCHSIM_TARGET"] = base
    from pytorchsim import config
    src = config.CONFIG_MACHINE_FILE
    with open(src) as f:
        machine = yaml.safe_load(f)
    machine["vpu_spad_size_kb_per_lane"] = budget_kb
    machine.setdefault("compiler", {})["spad_scoped_kb_per_lane"] = 0
    here = os.path.dirname(src)
    for key in ("ramulator_config_path", "booksim_config_path"):
        if key in machine:
            machine[key] = os.path.normpath(os.path.join(here, machine[key]))
    path = os.path.join(tempfile.mkdtemp(prefix="torchsim-machine-"), f"{base}_{budget_kb}kb.yml")
    with open(path, "w") as f:
        yaml.safe_dump(machine, f)
    os.environ["TORCHSIM_TARGET"] = path
    return path


class Outcomes(list):
    """(config kwargs, 'compiled' or the OutOfResources message) of each precompile of a config with `key`, in
    order; a context manager, needs TORCHINDUCTOR_COMPILE_THREADS=1 so the precompile runs in this process."""

    def __init__(self, key):
        super().__init__()
        self.key = key

    def __enter__(self):
        from torch._inductor.runtime import triton_heuristics as th
        from triton.runtime.errors import OutOfResources
        self.th, self.orig = th, th.CachingAutotuner._precompile_config
        outcomes = self

        def spy(this, cfg, **kw):
            if outcomes.key not in cfg.kwargs:
                return outcomes.orig(this, cfg, **kw)
            try:
                got = outcomes.orig(this, cfg, **kw)
            except OutOfResources as e:
                outcomes.append((dict(cfg.kwargs), f"OutOfResources: {e}"))
                raise
            outcomes.append((dict(cfg.kwargs), "compiled"))
            return got

        th.CachingAutotuner._precompile_config = spy
        return self

    def __exit__(self, *exc):
        self.th.CachingAutotuner._precompile_config = self.orig
