import time
import argparse
import sys
import math
import m5
from m5.objects import *

sys.path.append(os.environ.get('TORCHSIM_DIR'))
from gem5_script.vpu_config import *

bin_path = sys.argv[1]
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cmd", default="", help="The binary to run in syscall emulation mode.")
parser.add_argument("-o", "--options", default="", help="""The options to pass to the binary, use around the entire string""")
parser.add_argument("--cpu", choices=["RiscvAtomicSimpleCPU", "RiscvTimingSimpleCPU", "RiscvMinorCPU", "RiscvDerivO3CPU",
                                      "RiscvMinorCPU", "RiscvCustomCPU", "RiscvMinorV2CPU", "RiscvMinorV4CPU", "RiscvVPU",
                                      "RiscvSparseVPU"], default="RiscvVPU")
parser.add_argument("--mem", choices=["SimpleMemory", "ScratchpadMemory", "DDR3_1600_8x8"], default="ScratchpadMemory")
parser.add_argument("--sparse", type=bool, default=False)
parser.add_argument("--vlane", type=int, default=128)
parser.add_argument("--vlen", type=int, default=256)
args = parser.parse_args()

class InstMemory(SimpleMemory):
    latency = "1ns"
    bandwidth = "64GB/s"

class SpadMemory(SimpleMemory):
    latency = "1ns" # latency unit is "tick" 1ns = 1000 ticks

    def __init__(self, bandwidth="4GB/s"):
        super().__init__()
        self.bandwidth = bandwidth  # Set the bandwidth for this memory bank

class MultiBankMemorySystem():
    def __init__(self, bus_port, mem_range, num_banks=8, granule_size=4, total_bandwidth="32GB/s"):
        self.num_banks = num_banks
        self.granule_size = granule_size

        # Calculate interleaving properties
        self.intlvBits = int(math.log2(self.num_banks)) # Interleaving bits
        self.intlvLowBit = int(math.log2(self.granule_size))  # Granule size low bit
        self.intlvHighBit = self.intlvLowBit + self.intlvBits - 1  # High bit for interleaving
        self.mem_ctrls = []
        self.bandwidth_per_bank = self.divide_bandwidth(total_bandwidth[:-2], self.num_banks)

        # Create memory controllers for each bank
        self.create_memory_banks(bus_port, mem_range)

    def create_memory_banks(self, bus_port, mem_range):
        """Create memory banks and interleave them"""
        for i in range(self.num_banks):
            #print(f"[Spad Bank {i}] Bandwidth {self.bandwidth_per_bank}")
            mem = SpadMemory(self.bandwidth_per_bank)  # Create a new memory bank
            # Define the memory range for each bank (interleaving range)
            if self.num_banks!=1:
                print("intlvBits:",self.intlvBits, " intlvHighBits: ", self.intlvHighBit)
                mem.range = AddrRange(
                    start=mem_range.start, size=mem_range.size(),
                    intlvBits=self.intlvBits, intlvMatch=i, intlvHighBit=self.intlvHighBit
                )
            else:
                mem.range = AddrRange(start=mem_range.start, size=mem_range.size())
            mem.port = bus_port
            self.mem_ctrls.append(mem)

    def divide_bandwidth(self, total_bandwidth, num_banks):
        total_bandwidth_bytes = self.bandwidth_to_bytes(total_bandwidth)
        per_bank_bandwidth_bytes = total_bandwidth_bytes / num_banks
        return self.bytes_to_bandwidth(per_bank_bandwidth_bytes)

    def bandwidth_to_bytes(self, bandwidth):
        # Extract the value and unit
        value, unit = bandwidth[:-2], bandwidth[-2:]
        value = float(value)
        # Convert based on the unit
        if unit == "GB":
            return value * 1e9
        elif unit == "MB":
            return value * 1e6
        elif unit == "KB":
            return value * 1e3
        elif unit == "B":
            return value
        else:
            raise ValueError(f"Unknown bandwidth unit: {unit}")

    def bytes_to_bandwidth(self, bandwidth_bytes):
        if bandwidth_bytes >= 1e9:
            return f"{bandwidth_bytes / 1e9}GB/s"
        elif bandwidth_bytes >= 1e6:
            return f"{bandwidth_bytes / 1e6}MB/s"
        elif bandwidth_bytes >= 1e3:
            return f"{bandwidth_bytes / 1e3}KB/s"
        else:
            return f"{bandwidth_bytes}B/s"

    def get_ctrls(self):
        return self.mem_ctrls

class L1Cache(NoncoherentCache):
    """Simple L1 Cache with default values"""
    assoc = 8
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 16
    tgts_per_mshr = 20
    def connectBus(self, bus):
        self.mem_side = bus.cpu_side_ports

    def connectCPU(self, cpu):
        raise NotImplementedError

class L1ICache(L1Cache):
    size = "8192kB"
    tag_latency = 0
    data_latency = 0
    response_latency = 0

    def connectCPU(self, cpu):
        self.cpu_side = cpu.icache_port

valid_cpu = {
    "RiscvMinorCPU": RiscvMinorCPU,
    "RiscvDerivO3CPU": RiscvO3CPU,
    "RiscvMinorCPU": RiscvMinorCPU,
    "RiscvVPU": RiscvVPU,
}

# change systolicArrayWidth and systolicArrayHeight into args.vlane
SystolicArray.systolicArrayWidth = args.vlane
SystolicArray.systolicArrayHeight = args.vlane
binary = args.cmd

# Main System Setup
system = System()
system.workload = SEWorkload.init_compatible(binary)

# Clock setting
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "1GHz"
system.clk_domain.voltage_domain = VoltageDomain()

fast_clk = SrcClockDomain()
fast_clk.clock = '8GHz'
fast_clk.voltage_domain = VoltageDomain()

system.mem_mode = "timing"
system.cache_line_size = 64
system.cpu = valid_cpu[args.cpu]()
system.cpu.ArchISA.vlen = args.vlen

# Memory range
granule_sz = 64
spad_num_bank = 1
system.mem_ranges = [AddrRange(start=0, size="16GB")]

system.membus = SpmXBar(
        width = granule_sz,
        header_latency = 0,
        frontend_latency = 0,
        forward_latency = 0,
        response_latency = 0)
system.membus.clk_domain = fast_clk

# Instruction cache connection
system.cpu.icache= L1ICache()
system.cpu.icache.connectCPU(system.cpu)
system.cpu.icache.connectBus(system.membus)
#system.cpu.icache.mem_side = inst_mem.port
system.cpu.dcache_port = system.membus.cpu_side_ports
system.cpu.createInterruptController()

# Create and connect memory nodes
multi_banked_spm = MultiBankMemorySystem(system.membus.mem_side_ports, system.mem_ranges[0], num_banks=spad_num_bank, granule_size=granule_sz)
system.mem_ctrls = multi_banked_spm.get_ctrls()

system.system_port = system.membus.cpu_side_ports

process = Process()
process.cmd = [binary] + args.options.split()
system.cpu.workload = process
system.cpu.createThreads()

# Simulation
root = Root(full_system=False, system=system)
m5.instantiate()
start_time = time.time()
exit_event = m5.simulate()

if exit_event.getCause() != "exiting with last active thread context":
    exit(1)
end_time = time.time()
elapsed_seconds = end_time - start_time
print(f"Simulation time: {elapsed_seconds:.6f} seconds")
