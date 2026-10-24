"""What PyTorchSim plugs into torch.compile, one package per layer it meets:

    dynamo/      npu's DeviceInterface
    inductor/    npu as an Inductor backend and the patches to Inductor's passes, choices and codegen
    dispatcher/  ops with a Triton kernel of their own, registered at the npu dispatch keys

What compiles and runs a kernel after Inductor is pytorchsim/backend and pytorchsim/simulator.
"""


def install():
    """Plug npu into every layer, once."""
    global _installed
    if _installed:
        return
    from . import dispatcher, dynamo, inductor

    dynamo.install()
    inductor.install()
    dispatcher.install()
    _installed = True


_installed = False
