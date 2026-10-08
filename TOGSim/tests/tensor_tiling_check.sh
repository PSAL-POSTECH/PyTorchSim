#!/bin/bash
# Build and run tests/tensor_tiling_check.cc against a configured TOGSim build dir ($1, default
# ../build), with the flags it compiled Instruction.cc with. Run where that build was configured.
set -e
here=$(cd "$(dirname "$0")" && pwd); build=$(cd "${1:-$here/../build}" && pwd)
flags=$(python3 -c "import json,shlex,sys; c=[e for e in json.load(open(sys.argv[1])) if e['file'].endswith('src/Instruction.cc')][0]['command']; a=shlex.split(c); print(' '.join(shlex.quote(x) for x in a[1:] if x.startswith(('-I','-D','-std','-isystem')) or x.startswith('/root/.conan/data/zlib')))" "$build/compile_commands.json")
libs=$(tr ' ' '\n' < "$build/src/CMakeFiles/Simulator.dir/link.txt" | grep -E '^-L|^-Wl,-rpath' | tr '\n' ' ')
eval c++ $flags -O1 -o "$build/tensor_tiling_check" "$here/tensor_tiling_check.cc" "$here/../src/Instruction.cc" $libs -lspdlog -lfmt
"$build/tensor_tiling_check"
