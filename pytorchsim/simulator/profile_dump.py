"""TORCHSIM_PROFILE_DUMP=1: at exit, write <dump path>/profile/ (trace.pftrace, tiles/, index.html) from the
TOGSim logs this process started. scripts/dump_profile.py does the same from the command line."""
import atexit
import os
import sys

from pytorchsim import config

_LOGS = []
_INSTALLED = [False]
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS = os.path.join(_ROOT, "scripts")


def record_log(log_path):
    """Remember a TOGSim log of this process for the exit dump; installs the exit hook on the first one."""
    if not config.CONFIG_PROFILE_DUMP:
        return
    _LOGS.append(log_path)
    if not _INSTALLED[0]:
        _INSTALLED[0] = True
        atexit.register(_write_profile)


def _write_profile():
    """The dump of the recorded logs; a failure is reported on stderr and never changes the exit status."""
    sys.path.insert(0, _SCRIPTS)
    try:
        import dump_profile
        out = os.path.join(config.get_dump_path(), "profile")
        trace_dir = os.environ.get("TORCH_TRACE")
        result = dump_profile.dump(_LOGS, out, torch_trace=[trace_dir] if trace_dir else [])
    except Exception as exc:
        print(f"[profile] dump failed: {exc}", file=sys.stderr)
        return
    if result:
        print(f"[profile] wrote {out}/index.html", file=sys.stderr)
