"""npu's mm/bmm and addmm/baddbmm template heuristics, in Inductor's template heuristics registry
(torch._inductor.heuristics.template.registry); the heuristics are
autotune/template_kernel_configs.py's."""


def install():
    """Register the mm/bmm heuristic and its addmm/baddbmm form for npu."""
    from torch._inductor.kernel.bmm import bmm_template
    from torch._inductor.kernel.mm import mm_template
    from torch._inductor.heuristics.template.registry import register_template_heuristic

    from pytorchsim.frontend.inductor.autotune.template_kernel_configs import mm_config_heuristics

    mm, addmm = mm_config_heuristics()
    register_template_heuristic(mm_template.uid, "npu")(mm)
    register_template_heuristic(bmm_template.uid, "npu")(mm)
    register_template_heuristic(mm_template.uid, "npu", op_name="addmm")(addmm)
    register_template_heuristic(bmm_template.uid, "npu", op_name="baddbmm")(addmm)
