from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)


class Hardware:
    sa: int                 # systolic array dimension (see note in config.py)
    vlane: int              # vpu_num_lanes
    vlen_bits: int          # vpu_vector_length_bits
    elem_bits: int = 32

    @property
    def lane_elems(self) -> int:
        return self.vlen_bits // self.elem_bits

    @property
    def throughput(self) -> int:
        return self.vlane * self.lane_elems


def _round_down(value: int, multiple: int) -> int:
    return max(multiple, (value // multiple) * multiple)


def systolic_tile(m: int, n: int, k: int, hw: Hardware) -> tuple[int, int, int]:
    col_unit = hw.sa * hw.vlane
    tm = _round_down(min(m, 32), hw.sa)
    tn = _round_down(min(n, col_unit * 4), col_unit)
    tk = _round_down(min(k, 64), hw.sa)
    return tm, tn, tk


def check_systolic(n: int, hw: Hardware) -> None:
    col_unit = hw.sa * hw.vlane
    if n < col_unit:
        raise ValueError(
            f"{n} columns is under one matmul instruction (SA*VLANE = {col_unit}); "
            f"the array cannot be driven at this shape")


def reducing_tile(rows: int, cols: int, hw: Hardware) -> tuple[int, int]:
    tn = min(cols, max(hw.throughput, hw.lane_elems * hw.vlane))
    tm = max(1, min(rows, max(1, (hw.throughput * 8) // max(1, tn))))
    return tm, tn


def pointwise_tile(rows: int, cols: int, hw: Hardware) -> tuple[int, int]:
    budget = hw.throughput * 32
    tn = min(cols, budget)
    tm = max(1, min(rows, max(1, budget // max(1, tn))))
    return tm, tn
