"""How each npu kernel's sizes are chosen, by the kind of kernel:

    template_kernel_configs.py    the configs a template kernel (mm, bmm, addmm, baddbmm, conv) is offered
    template_kernel_selection.py  which offered config a template kernel takes
    generated_kernel_blocks.py    the blocks of a kernel Inductor generates from nodes (any non-template):
                                  pinned from the machine, persistence and split of its reductions
"""


def install():
    """Apply the template kernels' choices and selection to Inductor."""
    from . import template_kernel_configs, template_kernel_selection

    template_kernel_configs.install()
    template_kernel_selection.install()
