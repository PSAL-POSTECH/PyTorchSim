"""Compile cache for the codegen route: torchsim_compile(src, meta, kernel_name) turns one kernel into
a launcher the call site calls with the kernel's args; one directory per source hash holds the
compiler's kernel file and every artifact."""

import copy
import os

from filelock import FileLock
from torch._inductor.codecache import get_hash

from pytorchsim import config, contract
from pytorchsim.backend import retile, compiler_bridge, compiler_input, launch_hooks
from pytorchsim.backend.build_identity import cached_elf, store_identity
from pytorchsim.simulator import functional, timing, wallclock

logger = config.setup_logger()

LOCK_TIMEOUT = 600


def _write_path(src_code, meta):
    """The kernel's workdir, keyed by everything its ELF is built from: the source, and the
    constexprs, grid, arguments and graph_prefetch the spec bakes in (a held and a streamed
    build of one source differ only there)."""
    baked = (meta.get("signature"), meta.get("constants"), meta.get("fixed_config"),
             meta.get("template_grid"), meta["graph_prefetch"], meta.get("numels"),
             [(a["name"], a["role"], a["dtype"], a["numel"]) for a in meta.get("args", ())])
    return os.path.join(config.get_dump_path(),
                        "triton_" + get_hash(src_code.strip() + repr(baked))[1:12])


class TritonNPULauncher:
    """What a compiled kernel name is bound to in the generated wrapper: each call launches the whole
    grid, Spike first so the tensors hold real values even if TOGSim fails; Spike on the ELF built for
    the launch's extents, a new one per extents the import-time build (a dynamic graph's hints) missed."""
    def __init__(self, kernel_name, workdir, meta, src_code=None):
        self.kernel_name = kernel_name
        self.workdir = workdir
        self.meta = meta
        self.src_code = src_code
        self._builds = {}

    def __call__(self, *args, torchsim_arg_names=None):
        launch_hooks.notify_launch(self, args)
        if config.pytorchsim_functional_mode:
            build = self._get_build_for(args)
            with wallclock.span(wallclock.SPIKE, self.kernel_name):
                written = functional.run(build.workdir, build.meta, args)
            logger.info("[Spike] %s wrote %s", self.kernel_name, written)
        else:
            logger.warning(
                "[Spike] %s: functional mode is off, so the output tensors keep "
                "whatever they held", self.kernel_name)

        if not config.pytorchsim_timing_mode:
            logger.warning(
                "[timing] %s: timing mode is off, so no cycles are reported",
                self.kernel_name)
            return None

        result = timing.run(self.workdir, self.meta, args, torchsim_arg_names)
        if result is None:
            logger.info("[TOGSim] %s queued on the open TOGSimulator; the stream's cycles "
                        "are reported when the simulator closes", self.kernel_name)
        else:
            logger.info("[TOGSim] %s simulated: %d cycles", self.kernel_name, result["total_cycles"])
        return result

    def _get_build_for(self, args):
        """This launcher when the launch's extents are the ones it was built for, else the build for
        them: each tensor's storage span and each numel the launch passes as a scalar, baked in an ELF."""
        extents = _get_launch_extents(self.meta, args)
        if extents == _get_launch_extents(self.meta, None) or self.src_code is None:
            return self
        if extents not in self._builds:
            from pytorchsim.simulator.placement import tensor_args
            meta = copy.deepcopy(self.meta)
            for a, (_, t), (_, span) in zip(meta["args"], tensor_args(self.meta, args), extents[0]):
                a["numel"] = span
                if a.get("sizes") is not None:
                    a["sizes"] = list(t.shape)
            meta["numels"].update(dict(extents[1]))
            logger.info("[torchsim-compile] %s: a launch over %s, not the %s it was built for; building it",
                        self.kernel_name, extents, _get_launch_extents(self.meta, None))
            self._builds[extents] = torchsim_compile(self.src_code, meta, self.kernel_name)
        return self._builds[extents]


def _get_launch_extents(meta, args):
    """((arg, storage span) per tensor, (numel, value) per numel passed as a scalar) of a launch, or of the
    build itself when `args` is None."""
    from pytorchsim.contract import storage_span
    from pytorchsim.simulator.placement import tensor_args

    if args is None:
        spans = tuple((a["name"], a["numel"]) for a in meta["args"])
        return spans, tuple(sorted((k, v) for k, v in meta["numels"].items() if k in _get_runtime_params(meta)))
    spans = tuple((m["name"], storage_span(t.shape, t.stride())) for m, t in tensor_args(meta, args))
    bound = dict(zip(_get_runtime_params(meta), args))
    numels = tuple(sorted((k, bound[k]) for k in meta["numels"] if k in bound
                          and isinstance(bound[k], int) and not isinstance(bound[k], bool)))
    return spans, numels


def _get_runtime_params(meta):
    """The kernel's parameters a launch passes, in order: the signature's that are not constexprs."""
    return [k for k, v in (meta.get("signature") or {}).items() if v != "constexpr"]


def torchsim_compile(src_code, meta, kernel_name):
    """Compile one Inductor-generated Triton kernel through the compiler, synchronously, when the
    generated wrapper is imported, holding what meta's graph_prefetch states; the compiler's
    refusal to hold it is an error: the lowering's election and the compiler disagree."""
    write_path = _write_path(src_code, meta)
    os.makedirs(write_path, exist_ok=True)

    lock = FileLock(os.path.join(write_path, ".compile.lock"), timeout=LOCK_TIMEOUT)
    with lock:
        spec_path = os.path.join(write_path, f"{kernel_name}_spec.py")
        elf = cached_elf(write_path, kernel_name)
        if (elf is not None and config.pytorchsim_timing_mode
                and contract.artifact(write_path, "tog_so") is None):
            logger.info("[torchsim-compile] %s: cached without --tog, rebuilding for timing",
                        kernel_name)
            elf = None
        if elf is None:
            with open(os.path.join(write_path, "kernel.py"), "w") as f:
                f.write(src_code)
            compiler_input.store_meta(write_path, meta)
            retries = retile.smaller_tiles(src_code, meta)
            while True:
                compiler_input.write(src_code, meta, spec_path)
                try:
                    with wallclock.span(wallclock.TORCHSIM_COMPILE, kernel_name):
                        compiler_bridge.run_pipeline(
                            spec_path, write_path, to_stage="pytorchsim-compile",
                            tog=bool(config.pytorchsim_timing_mode))
                    wallclock.ingest_compile(write_path, kernel_name)
                    break
                except compiler_bridge.CompilerError as exc:
                    _refuse_held(exc, meta)
                    why = retile.too_big(exc)
                    if why is None:
                        raise
                    retry = next(retries, None)
                    if retry is None:
                        logger.warning("[torchsim-compile] %s: %s, and no smaller tile is "
                                       "left to try", kernel_name, why)
                        raise
                    tile, src_code, tuned = retry
                    meta.update(tuned)
                    logger.info("[torchsim-compile] %s: %s, trying %s", kernel_name, why, tile)
            with open(os.path.join(write_path, "kernel.py"), "w") as f:
                f.write(src_code)
            compiler_input.store_meta(write_path, meta)
            store_identity(write_path)
        else:
            compiler_input.load_tuned_blocks(write_path, meta)
        logger.info("[torchsim-compile] %s -> %s", kernel_name, write_path)
        return TritonNPULauncher(kernel_name, write_path, meta, src_code)


def _refuse_held(exc, meta):
    """Raise, naming the kernel and the held args, when the compiler refused to hold what meta's
    graph_prefetch states (contract/failure): a scratchpad overflow or no prefetch candidate."""
    got = getattr(exc, "failure", None) or {}
    kinds = compiler_bridge.compiler_contract("failure")
    if meta["graph_prefetch"] and got.get("kind") in (kinds.SPAD_OVERFLOW, kinds.NOT_A_PREFETCH_CANDIDATE):
        raise compiler_bridge.CompilerError(
            f"{meta['kernel_name']}: the compiler refuses to hold {meta['graph_prefetch']} "
            f"({got['kind']}: {got.get('message')}), which the lowering elected", exc.cmd, got) from exc
