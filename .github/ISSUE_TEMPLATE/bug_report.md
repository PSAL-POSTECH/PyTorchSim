---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: bug
assignees: YWHyuk

---

**Describe the bug**
A clear and concise description of what the bug is.
If applicable, add screenshots to help explain your problem.

**To Reproduce**
If the issue occurs while running a Python workload or involves a simulator crash, please also provide:
- The Python script and any relevant configuration files (e.g., extension_config.py).
- For simulator crashes, the exact simulator command and arguments used for execution.
- The directory containing the generated wrapper code and binaries.

For example:
```
python3 tests/ops/elementwise/test_add.py
...
[SpikeSimulator] cmd> spike --isa rv64gcv --varch=vlen:256,elen:64 --vectorlane-size=128 \
  -m0x80000000:0x1900000000,0x2000000000:0x1000000 \
  --scratchpad-base-paddr=137438953472 --scratchpad-base-vaddr=3489660928 --scratchpad-size=131072 \
  --kernel-addr=0000000000010404:10506 \
  --base-path=/tmp/torchinductor/tmp/g3smoqaa2r5/runtime_0000 \
  /workspace/riscv-pk/build/pk \
  /tmp/torchinductor/tmp/g3smoqaa2r5/validation_binary \
  /tmp/torchinductor/tmp/g3smoqaa2r5/runtime_0000/arg0_1/0.raw \
  /tmp/torchinductor/tmp/g3smoqaa2r5/runtime_0000/arg1_1/0.raw \
  /tmp/torchinductor/tmp/g3smoqaa2r5/runtime_0000/buf0/0.raw
```
If a crash occurs, please compress and attach the related directory (e.g. '/tmp/torchinductor/tmp/g3smoqaa2r5`) to help reproduce the issue.
