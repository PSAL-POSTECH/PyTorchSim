"""What the stages share, so none of them imports another for it; nothing here imports torch or the compiler:

    launch          the axes a kernel's programs spread over, their extents and the grid
    storage         how far a strided layout reaches
    kernel object   the compiler's manifest in a workdir and the artifacts it names
    launch dir      each process's corner of a shared workdir
"""

import json
import os


class SpecIncomplete(RuntimeError):
    """Metadata the compiler requires that this kernel did not provide, with the missing field named."""


class KernelObjectMismatch(RuntimeError):
    """The kernel object in a workdir is not in the format this reader knows."""


# launch: the axes a kernel's programs spread over, their extents and the grid

_PARALLEL_PREFIXES = ("z", "y", "x")


def block_name(prefix):
    return f"{prefix.upper()}BLOCK"


def parallel_axes(numels):
    """Grid axes this kernel uses, outermost first."""
    return [p for p in _PARALLEL_PREFIXES if f"{p}numel" in numels]


def launch_axes(meta):
    """The axes the launch spreads programs over, outermost first; a template states all three, even at
    extent 1, as its source reads tl.program_id(0..2)."""
    if meta.get("template_grid") is None:
        return parallel_axes(meta["numels"])
    return list(_PARALLEL_PREFIXES)


def launch_extents(meta):
    """The extents of `launch_axes`, in the same order."""
    grid = meta.get("template_grid")
    if grid is None:
        return grid_of(meta)
    by_axis = dict(zip(("x", "y", "z"), grid))
    return tuple(by_axis[p] for p in launch_axes(meta))


def reduction_axes(numels):
    """Reduction prefixes, read off the keys (with their trailing underscore: "r0_" of `r0_numel`)."""
    return sorted(k[:-len("numel")] for k in numels
                  if k.endswith("numel") and k.startswith("r"))


def grid_of(meta):
    """Launch grid from the numels and the pinned blocks, outermost first; the timing path reads it with
    the launch's numels. A template states its own grid."""
    if meta.get("template_grid") is not None:
        return launch_extents(meta)

    numels = meta["numels"]
    cfg = meta.get("fixed_config") or {}
    axes = parallel_axes(numels)
    if not axes:
        raise SpecIncomplete(
            f"{meta['kernel_name']} has no parallel iteration axis to grid over")

    grid = []
    for prefix in axes:
        n, block = numels.get(f"{prefix}numel"), cfg.get(block_name(prefix))
        if n is None or not block:
            raise SpecIncomplete(
                f"cannot compute the grid for {meta['kernel_name']} axis "
                f"'{prefix}': {prefix}numel={n!r}, {block_name(prefix)}={block!r}. "
                f"Inductor defers the grid to triton_heuristics at runtime; this "
                f"route needs it statically (see fixed_config_for).")
        grid.append(-(-int(n) // int(block)))
    return tuple(grid)


def grid_xyz(meta):
    """The same grid in the compiler's order, (gridX, gridY, gridZ), built by axis name."""
    extents = dict(zip(launch_axes(meta), launch_extents(meta)))
    return tuple(extents.get(p, 1) for p in ("x", "y", "z"))


# storage: how far a strided layout reaches

def storage_span(size, stride, offset=0):
    """Last addressable element of a strided layout, in elements: past the element count when the
    strides leave gaps."""
    return offset + 1 + sum((s - 1) * t for s, t in zip(size, stride))


# kernel object: the compiler's manifest in a workdir and the artifacts it names

#: The compiler's manifest; its schema is the compiler's (contract/kernel_object.py), and KERNEL_FORMAT is
#: what stops this reader drifting from it.
KERNEL_MANIFEST = "kernel.json"


KERNEL_FORMAT = 9


def kernel_object(workdir):
    """The manifest the compiler left in `workdir`, or None if it did not get that far."""
    path = os.path.join(workdir, KERNEL_MANIFEST)
    try:
        with open(path) as fh:
            got = json.load(fh)
    except (OSError, ValueError):
        return None
    if got.get("format") != KERNEL_FORMAT:
        raise KernelObjectMismatch(
            f"{path}: kernel object format {got.get('format')!r}, this reader "
            f"knows {KERNEL_FORMAT}. Another compiler or frontend wrote it; "
            f"delete the workdir (scripts/clear_codegen_cache.sh) to rebuild it.")
    return got


def attribute_sources(manifest):
    """The args the kernel's runtime attributes are read from, by the manifest's `sources`, in arg order:
    what a run must hold as the launch did to record the launch's attributes. An attribute the manifest
    gives no `sources` is a KernelObjectMismatch."""
    attributes = (manifest.get("runtime_attributes") or {}).get("attributes", [])
    unsaid = [a["id"] for a in attributes if "sources" not in a]
    if unsaid:
        raise KernelObjectMismatch(
            f"{manifest['name']}: runtime attributes {unsaid} name no sources; the compiler that wrote "
            f"this kernel object does not say which args they are read from")
    named = {n for a in attributes for n in a["sources"]}
    unknown = sorted(named - {a["name"] for a in manifest["args"]})
    if unknown:
        raise KernelObjectMismatch(f"{manifest['name']}: runtime attribute sources {unknown} are no arg of it")
    return [a["name"] for a in manifest["args"] if a["name"] in named]


def artifact(workdir, kind):
    """One compiler output by the name the manifest gives it, or None; stage numbers are no interface."""
    got = kernel_object(workdir)
    if got is None:
        return None
    rel = got.get("artifacts", {}).get(kind)
    if not rel:
        return None
    full = os.path.join(workdir, rel)
    return full if os.path.exists(full) else None


# launch dir: each process's corner of a shared workdir

LAUNCH_DIR_PREFIX = "launch"


def dir_for(workdir):
    """Create and return this process's launch directory inside `workdir`."""
    d = os.path.join(workdir, f"{LAUNCH_DIR_PREFIX}-{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    return d


def runtime_dir(workdir):
    """Create and return this process's directory of `.raw` launch tensors."""
    d = os.path.join(dir_for(workdir), "runtime")
    os.makedirs(d, exist_ok=True)
    return d


def link_shared(workdir, names):
    """Relink `names` from the workdir into this launch directory, where TOGSim finds the cycle table and
    the shape file beside the TOG it is given. Returns it."""
    d = dir_for(workdir)
    for name in names:
        src, dst = os.path.join(workdir, name), os.path.join(d, name)
        if os.path.lexists(dst):
            os.remove(dst)
        if os.path.exists(src):
            os.symlink(os.path.relpath(src, d), dst)
    return d
