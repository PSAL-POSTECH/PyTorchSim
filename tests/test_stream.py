import torch
import time

start_event = torch.npu.event(enable_timing=True)
end_event = torch.npu.event(enable_timing=True)
stream = torch.npu.stream()

def my_kernel():
    print("Task is running...")
    result = sum(range(1000))
    time.sleep(2.5)
    print(f"Task completed with result: {result}")

start_event.record(stream)
stream.launch_kernel(my_kernel)
end_event.record(stream)


stream.synchronize()

elapsed_time = end_event.elapsed_time(start_event)
print("Event has completed! ", elapsed_time)