"""How far a strided layout reaches in storage.

Asked on both sides of the route -- the compile side sizes the buffer the binary
is handed, the runtime side sizes the .raw file that fills it -- so it is stated
once here rather than derived twice.
"""

def storage_span(size, stride, offset=0):
    """Last addressable element of a strided layout, in elements.

    NOT the product of `size`. The kernel addresses with the strides, and a
    layout whose strides leave gaps reaches past its element count.
    """
    return offset + 1 + sum((s - 1) * t for s, t in zip(size, stride))
