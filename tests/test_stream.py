import torch
import time

def my_kernel():
    print("Task is running...")
    result = sum(range(1000))
    time.sleep(2.5)
    print(f"Task completed with result: {result}")

torch.npu.launch_kernel(my_kernel)
torch.npu.synchronize()
print("Task completed!")