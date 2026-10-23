"""What PyTorchSim plugs into torch.compile, one package per layer it meets:

    inductor/    npu as an Inductor backend (its DeviceInterface among it) and the patches to
                 Inductor's passes, choices and codegen
    dispatcher/  ops with a Triton kernel of their own, registered at the npu dispatch keys
    tiling.py    the block and mm tile sizes this machine takes, which both read

What compiles and runs a kernel after Inductor is pytorchsim/backend and pytorchsim/simulator.
"""


def install():
    """Plug npu into every layer, once."""
    global _installed
    if _installed:
        return
    from . import dispatcher, inductor

    inductor.install()
    dispatcher.install()
    _installed = True


_installed = False
