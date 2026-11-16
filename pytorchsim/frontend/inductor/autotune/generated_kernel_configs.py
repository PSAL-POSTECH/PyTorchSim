"""The configs Inductor's pointwise and reduction heuristics hand a generated npu kernel: the candidate tiles
codegen put in inductor_meta (tiling.candidate_blocks), largest first; registered for npu in registration.py."""

from torch._inductor.heuristics.registry import CodegenConfigHeuristics

CANDIDATES_KEY = "pytorchsim_candidates"


def _configs(inductor_meta):
    """One triton.Config per candidate tile of the kernel, in the order codegen stated them."""
    import triton

    return [triton.Config(dict(c), num_warps=1, num_stages=1) for c in inductor_meta[CANDIDATES_KEY]]


class NPUPointwiseConfigs(CodegenConfigHeuristics):
    def get_configs(self, size_hints, bs, triton_config, hinted_configs, tile_hint=None,
                    inductor_meta=None):
        return _configs(inductor_meta)


class NPUReductionConfigs(CodegenConfigHeuristics):
    def get_configs(self, size_hints, inductor_meta, triton_meta, num_dynamic=0):
        return _configs(inductor_meta)

    def get_persistent_configs(self, size_hints, reduction_hint, inductor_meta, triton_meta):
        return _configs(inductor_meta)
