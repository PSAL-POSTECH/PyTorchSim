"""How each npu kernel's sizes are chosen, by the kind of kernel:

    template_kernel_configs.py    the configs a template kernel (mm, bmm, addmm, baddbmm, conv) is offered
    template_kernel_selection.py  which offered config a template kernel takes
    generated_kernel_blocks.py    the blocks of a kernel Inductor generates from nodes (any non-template):
                                  pinned from the machine, persistence and split of its reductions
"""


def install():
    """Apply the patches Inductor has no hook for (the conv's groups and grid, template selection);
    NPUChoices and the mm heuristics, which have hooks, are registration.py's."""
    from . import template_kernel_configs, template_kernel_selection

    template_kernel_configs.install()
    template_kernel_selection.install()
