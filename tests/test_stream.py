import torch
import time
from PyTorchSimFrontend.extension_codecache import (
    set_current_stream,
    get_current_stream,
    get_stream_index,
)


def my_kernel(tag):
    print(f"[{tag}] task is running...")
    result = sum(range(1000))
    time.sleep(0.2)
    print(f"[{tag}] task completed with result: {result}")

# Create two streams and register one as global stream for dummy_simulator fallback
stream0 = torch.npu.Stream()
stream1 = torch.npu.Stream()
set_current_stream(stream0)

print("Global stream is stream0:", get_current_stream() is stream0)
print("stream0 index:", get_stream_index(stream0))
print("stream1 index:", get_stream_index(stream1))
print("global stream index:", get_stream_index(get_current_stream()))

# Queue tasks on each stream
stream0.launch_kernel(lambda: my_kernel("stream0"))
stream1.launch_kernel(lambda: my_kernel("stream1"))

# Default API path (uses NPU default stream)
torch.npu.launch_kernel(lambda: my_kernel("default_stream"))
torch.npu.synchronize()
print("All stream tasks completed!")
