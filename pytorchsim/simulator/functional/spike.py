"""Spike: the compiler's launcher (its runtime/spike.py) run over a launch's whole grid, under the
options the kernel object was compiled for, with this environment's spike and pk."""

import os

from pytorchsim.backend import compiler_bridge


_VCIX_ENV = os.environ.get("VCIX_ENV_ROOT", "/opt/vcix-env")


SPIKE = os.environ.get("TORCHSIM_SPIKE", os.environ.get(
    "TORCHSIM_COMPILE_SPIKE", os.path.join(_VCIX_ENV, "spike", "bin", "spike")))


PK = os.environ.get("TORCHSIM_PK", os.environ.get(
    "TORCHSIM_COMPILE_PK", os.path.join(_VCIX_ENV, "pk", "riscv64-unknown-elf", "bin", "pk")))


def launch(obj_dir, manifest, runtime, raw_paths, log, scalar_values=None):
    """Run the whole launch grid with the compiler's launcher, under the options the object was
    compiled for, leaving each written argument in its .raw."""
    ko = compiler_bridge.compiler_contract("kernel_object")
    try:
        elf = ko.artifact(obj_dir, "elf", manifest=manifest)
        run = compiler_bridge.run_options(obj_dir, manifest)
    except (ko.ObjectError, compiler_bridge.NoTarget) as e:
        raise launcher().SpikeError(str(e)) from None
    want = os.environ.get("TORCHSIM_SPIKE_JOBS", os.environ.get("TORCHSIM_COMPILE_SPIKE_JOBS", ""))
    return launcher().launch(elf, manifest, runtime, raw_paths, run, spike=SPIKE, pk=PK, log=log,
                             want_jobs=want.strip(), scalar_values=scalar_values)


def launcher():
    """The compiler's Spike launcher module, from the checkout TORCHSIM_COMPILE_DIR names."""
    return compiler_bridge.compiler_module("runtime.spike")
