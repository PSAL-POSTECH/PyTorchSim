"""The gem5 half of the TOGSim trace: per-tile cycles into a cycle table.

The producer itself is the compiler's, C++ and .so both.
"""
import os


def cycle_table(overlap_offsets, cycle_list):
    """[(cycle, overlapping_cycle), ...] indexed by tile_id. The compiler states each tile's
    overlap offset (None where nothing of it overlaps); gem5 gives the cycles."""
    if len(cycle_list) != len(overlap_offsets):
        raise ValueError(
            f"gem5 returned {len(cycle_list)} cycle sample(s) for "
            f"{len(overlap_offsets)} compute tile(s): a marker fired a different "
            f"number of times than there are tiles, so the table would be keyed "
            f"by the wrong samples")
    return [(int(c), 0 if off is None else max(int(c) - int(off), 0))
            for c, off in zip(cycle_list, overlap_offsets)]


def dump_cycle_table_tsv(table, path, origins=None):
    """cycle<TAB>overlapping per line, in tile_id order, for TOGSim's loader.

    origins is appended as a trailing comment line, which the loader stops at.
    """
    with open(path, "w") as fh:
        for cycle, overlapping in table:
            fh.write("%d\t%d\n" % (int(cycle), int(overlapping)))
        if origins:
            fh.write("# origins: %s\n" % ", ".join(sorted(str(o) for o in origins)))
    return path


def default_include_dir():
    """Where togsim_runtime.h lives. provenance hashes it into the cache key."""
    root = os.environ.get("TORCHSIM_DIR")
    if not root:
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
    return os.path.join(root, "TOGSim", "include")
