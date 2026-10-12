"""This process's private corner of a kernel workdir other runs also use.

The workdir caches what compiling produced -- a function of the source and the
machine, so sharing it is the point. A launch's tensors and grid are not.
"""

import os

PREFIX = "launch"


def dir_for(workdir):
    """Create and return this process's launch directory inside `workdir`."""
    d = os.path.join(workdir, f"{PREFIX}-{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    return d


def runtime_dir(workdir):
    """Create and return this process's directory of `.raw` launch tensors."""
    d = os.path.join(dir_for(workdir), "runtime")
    os.makedirs(d, exist_ok=True)
    return d


def link_shared(workdir, names):
    """Relink `names` from the workdir into this launch directory. Returns it.

    TOGSim resolves the cycle table and the shape file from the directory of
    the trace it is given, so the shared trace has to appear in this one.
    """
    d = dir_for(workdir)
    for name in names:
        src, dst = os.path.join(workdir, name), os.path.join(d, name)
        if os.path.lexists(dst):
            os.remove(dst)
        if os.path.exists(src):
            os.symlink(os.path.relpath(src, d), dst)
    return d
