"""What every fused kernel of our own shares: the scratchpad budget its plan is held to."""

from pytorchsim import config


def next_pow2(n):
    """The least power of two at or above `n` (1 for 0)."""
    return 1 << max(0, (int(n) - 1).bit_length())


def lane_bytes(tiles):
    """An upper bound on the bytes per lane the [rows, cols] tiles take: each in its own buffer (no
    reuse), banked across whichever axis leaves the most per lane. The compiler picks the axis
    (a decode's [GP, D] accumulator lands on its GP rows, its [GP, BKV] scores on BKV), so the
    bound takes the shorter one, at most the lane count."""
    lanes = int(config.CONFIG_MACHINE.vector_lanes)
    return sum(rows * cols * nbytes // min(rows, cols, lanes) for rows, cols, nbytes in tiles)


def fits_scratchpad(tiles):
    """Whether `tiles` fit one lane's scratchpad budget, and each is a tensor Triton builds."""
    from pytorchsim.frontend.inductor.codegen import tiling
    big = tiling.triton_max_numel()
    return (all(r * c <= big for r, c, _ in tiles)
            and lane_bytes(tiles) <= int(config.CONFIG_MACHINE.spad_budget))
