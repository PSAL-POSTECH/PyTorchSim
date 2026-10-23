"""Everything npu registers with Inductor, one file per registry it fills, in install()'s order:

    device_interface.py     Dynamo's device interfaces: what Inductor asks of the device
    triton_gpu.py           GPU_TYPES (a patch: it has no hook)
    device_codegen.py       Inductor's device codegens: scheduling, wrapper, post-grad pass
    device_op_overrides.py  Inductor's device op overrides: the device code in a wrapper
    choices.py              Inductor's choices class: NPUChoices, answered by autotune/
    template_heuristics.py  Inductor's template heuristics: the mm/bmm configs, autotune/'s
"""


def install():
    """Register npu in every registry: the device interface first, since Inductor's Triton check
    (has_triton) asks it."""
    from . import (choices, device_codegen, device_interface, device_op_overrides,
                   template_heuristics, triton_gpu)

    device_interface.install()
    triton_gpu.install()
    device_codegen.install()
    device_op_overrides.install()
    choices.install()
    template_heuristics.install()
