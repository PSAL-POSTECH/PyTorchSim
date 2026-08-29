from __future__ import annotations

import os
import re
import subprocess


def _assignment(name: str) -> re.Pattern:
    return re.compile(
        rf"^(\s*static\s+const\s+\w+(?:_t)?\s+{re.escape(name)}\s*=\s*)([^;]+)(;)",
        re.MULTILINE)


def retarget(source: str, values: dict[str, int]) -> str:
    text = source
    for name, value in values.items():
        pattern = _assignment(name)
        text, n = pattern.subn(rf"\g<1>{int(value)}\g<3>", text, count=1)
        if n == 0:
            raise KeyError(
                f"constant {name!r} is not declared in this kernel; the tile "
                f"names in tiling.py no longer match the .cpp")
    return text


def emit(kernel_path: str, values: dict[str, int], out_cpp: str) -> str:
    with open(kernel_path) as fh:
        source = fh.read()
    os.makedirs(os.path.dirname(out_cpp) or ".", exist_ok=True)
    with open(out_cpp, "w") as fh:
        fh.write(retarget(source, values))
    return out_cpp


def compile_so(cpp_path: str, include_dir: str, out_so: str) -> str:
    if os.path.exists(out_so) and \
            os.path.getmtime(out_so) >= os.path.getmtime(cpp_path):
        return out_so
    subprocess.run(
        ["g++", "-shared", "-fPIC", "-std=gnu++17", "-O0",
         "-I", include_dir, cpp_path, "-o", out_so],
        check=True, capture_output=True, text=True)
    return out_so
