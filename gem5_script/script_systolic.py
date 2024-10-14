import argparse
import os
import sys

import m5
from m5.objects import *

from ctypes import cdll

bin_path = sys.argv[1]
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--cmd",
    default="",
    help="The binary to run in syscall emulation mode.",
)
parser.add_argument(
    "-o",
    "--options",
    default="",
    help="""The options to pass to the binary, use
                            around the entire string""",
)

class MySimpleMemory(SimpleMemory):
    latency = "1ns"

class SpadMemory(SimpleMemory):
    latency = "1ns" # latency unit is "tick" 1ns = 1000 ticks
    bandwidth = "64GB/s"
    # TODO: bandwidth = "XXGB/s" what is a proper value? (ref. simple_mem.cc:154)

class SystolicArray(MinorFU):
    unitType = "SystolicArray"
    opClasses = minorMakeOpClassSet([
        "CustomMatMul",
        "CustomMatMuliVpush",
        "CustomMatMulwVpush",
        "CustomMatMulvpop",
        ])
    opLat = 1
    systolicArrayWidth = 128
    systolicArrayHeight = 128

class SparseAccelerator(MinorFU):
    unitType = "SparseAccelerator"
    opClasses = minorMakeOpClassSet([
        "CustomMatMul",
        "CustomMatMuliVpush",
        "CustomMatMulwVpush",
        "CustomMatMulvpop",
        ])
    opLat = 1

class SpecialFunctionUnit(MinorFU):
    opClasses = minorMakeOpClassSet([
        "CustomMatMulvexp",
        ])
    opLat = 10

class MinorVecAdder(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdAdd",
            "SimdFloatAdd",
            "SimdFloatAlu",
            "SimdFloatCmp",
        ]
    )
    opLat = 1

class MinorVecMultiplier(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdMult",
            "SimdFloatMult",
        ]
    )
    opLat = 3

class MinorVecDivider(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdDiv",
            "SimdFloatDiv",
        ]
    )
    opLat = 5

class MinorVecMisc(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdUnitStrideLoad",
            "SimdUnitStrideStore",
            "SimdUnitStrideMaskLoad",
            "SimdUnitStrideMaskStore",
            "SimdStridedLoad",
            "SimdStridedStore",
            "SimdIndexedLoad",
            "SimdIndexedStore",
            "SimdUnitStrideFaultOnlyFirstLoad",
            "SimdWholeRegisterLoad",
            "SimdWholeRegisterStore",
            "SimdAddAcc",
            "SimdAlu",
            "SimdCmp",
            "SimdCvt",
            "SimdMultAcc",
            "SimdMatMultAcc",
            "SimdShift",
            "SimdShiftAcc",
            "SimdSqrt",
            "SimdFloatCvt",
            "SimdFloatMisc",
            "SimdFloatMultAcc",
            "SimdFloatMatMultAcc",
            "SimdFloatSqrt",
            "SimdReduceAdd",
            "SimdReduceAlu",
            "SimdReduceCmp",
            "SimdFloatReduceAdd",
            "SimdFloatReduceCmp",
            "SimdAes",
            "SimdAesMix",
            "SimdSha1Hash",
            "SimdSha1Hash2",
            "SimdSha256Hash",
            "SimdSha256Hash2",
            "SimdShaSigma2",
            "SimdShaSigma3",
            "SimdPredAlu",
            "SimdMisc",

            "SimdUnitStrideSegmentedLoad",
            "SimdUnitStrideSegmentedStore",
            "SimdExt",
            "SimdFloatExt",
        ]
    )
    opLat = 1

class MinorVecConfig(MinorFU):
    opClasses = minorMakeOpClassSet(
        [    
            "SimdConfig",
        ]
    )
    opLat = 1

class MinorCustomVecFU(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdUnitStrideLoad",
            "SimdUnitStrideStore",
            "SimdUnitStrideMaskLoad",
            "SimdUnitStrideMaskStore",
            "SimdStridedLoad",
            "SimdStridedStore",
            "SimdIndexedLoad",
            "SimdIndexedStore",
            "SimdUnitStrideFaultOnlyFirstLoad",
            "SimdWholeRegisterLoad",
            "SimdWholeRegisterStore",
            "SimdAdd",
            "SimdAddAcc",
            "SimdAlu",
            "SimdCmp",
            "SimdCvt",
            "SimdMisc",
            "SimdMult",
            "SimdMultAcc",
            "SimdMatMultAcc",
            "SimdShift",
            "SimdShiftAcc",
            "SimdDiv",
            "SimdSqrt",
            "SimdFloatAdd",
            "SimdFloatAlu",
            "SimdFloatCmp",
            "SimdFloatCvt",
            "SimdFloatDiv",
            "SimdFloatMisc",
            "SimdFloatMult",
            "SimdFloatMultAcc",
            "SimdFloatMatMultAcc",
            "SimdFloatSqrt",
            "SimdReduceAdd",
            "SimdReduceAlu",
            "SimdReduceCmp",
            "SimdFloatReduceAdd",
            "SimdFloatReduceCmp",
            "SimdAes",
            "SimdAesMix",
            "SimdSha1Hash",
            "SimdSha1Hash2",
            "SimdSha256Hash",
            "SimdSha256Hash2",
            "SimdShaSigma2",
            "SimdShaSigma3",
            "SimdPredAlu",
            "SimdMisc",
            "SimdConfig",
        ]
    )
    opLat = 1

class MinorCustomIntFU(MinorFU):
    opClasses = minorMakeOpClassSet(["IntAlu"])
    timings = [MinorFUTiming(description="Int", srcRegsRelativeLats=[2])]
    opLat = 1

class MinorCustomFUPool(MinorFUPool):
    funcUnits = [         
        SystolicArray(), # 0
        
        MinorVecConfig(), # 1 for vector config   

        MinorVecMisc(), # 2~5
        MinorVecMisc(),
        MinorVecMisc(),
        MinorVecMisc(),

        # ALU0
        MinorVecAdder(), # 6
        MinorVecMultiplier(), # 7
        MinorVecDivider(), # 8
        MinorVecAdder(), # 9
        MinorVecMultiplier(), # 10
        MinorVecDivider(), # 11
        MinorVecAdder(), # 12
        MinorVecMultiplier(), # 13
        MinorVecDivider(), # 14
        MinorVecAdder(), # 15
        MinorVecMultiplier(), # 16
        MinorVecDivider(), # 17

        # ALU1
        MinorVecAdder(), # 18 ~ 29
        MinorVecMultiplier(),
        MinorVecDivider(),
        MinorVecAdder(),
        MinorVecMultiplier(),
        MinorVecDivider(),
        MinorVecAdder(),
        MinorVecMultiplier(),
        MinorVecDivider(),
        MinorVecAdder(),
        MinorVecMultiplier(),
        MinorVecDivider(),

        MinorCustomIntFU(), # 30
        MinorCustomIntFU(),

        MinorDefaultIntMulFU(),
        MinorDefaultIntDivFU(),
        MinorDefaultPredFU(),
        MinorDefaultMemFU(),
        MinorDefaultMiscFU(),

        SpecialFunctionUnit(),

        # SparseAccelerator(),
        # Serializer0(),
        # Serializer1(),
        # DeSerializer(),
    ]

class MinorCustomSparseFUPool(MinorFUPool):
    funcUnits = [         
        MinorVecConfig(), # for vector config   

        MinorVecMisc(),
        MinorVecMisc(),
        MinorVecMisc(),
        MinorVecMisc(),

        # ALU0
        MinorVecAdder(),
        MinorVecMultiplier(),
        MinorVecDivider(),
        MinorVecAdder(),
        MinorVecMultiplier(),
        MinorVecDivider(),
        MinorVecAdder(),
        MinorVecMultiplier(),
        MinorVecDivider(),
        MinorVecAdder(),
        MinorVecMultiplier(),
        MinorVecDivider(),

        # ALU1
        MinorVecAdder(),
        MinorVecMultiplier(),
        MinorVecDivider(),
        MinorVecAdder(),
        MinorVecMultiplier(),
        MinorVecDivider(),
        MinorVecAdder(),
        MinorVecMultiplier(),
        MinorVecDivider(),
        MinorVecAdder(),
        MinorVecMultiplier(),
        MinorVecDivider(),

        MinorCustomIntFU(),
        MinorCustomIntFU(),

        MinorDefaultIntMulFU(),
        MinorDefaultIntDivFU(),
        MinorDefaultPredFU(),
        MinorDefaultMemFU(),
        MinorDefaultMiscFU(),

        SparseAccelerator(),
        # Serializer0(),
        # Serializer1(),
        # DeSerializer(),
    ]

class RiscvCustomCPU(RiscvMinorCPU):
    fetch2InputBufferSize = 4
    decodeInputWidth = 4
    executeInputWidth = 4
    executeIssueLimit = 8
    executeMemoryIssueLimit = 2
    executeCommitLimit = 8
    executeMemoryCommitLimit = 2
    executeFuncUnits = MinorCustomFUPool()

class RiscvVPU(RiscvMinorCPU):
    fetch2InputBufferSize = 2
    decodeInputBufferSize = 1
    decodeInputWidth = 1
    executeInputWidth = 8
    executeIssueLimit = 8
    executeMemoryIssueLimit = 8
    executeCommitLimit = 8
    executeMemoryCommitLimit = 8
    executeFuncUnits = MinorCustomFUPool()

class RiscvSparseVPU(RiscvMinorCPU):
    fetch2InputBufferSize = 2
    decodeInputBufferSize = 1
    decodeInputWidth = 1
    executeInputWidth = 8
    executeIssueLimit = 8
    executeMemoryIssueLimit = 8
    executeCommitLimit = 8
    executeMemoryCommitLimit = 8
    executeFuncUnits = MinorCustomSparseFUPool()

class MinorV2FUPool(MinorFUPool):
    funcUnits = [
        MinorDefaultIntFU(),
        MinorDefaultIntFU(),
        MinorDefaultIntMulFU(),
        MinorDefaultIntDivFU(),
        MinorDefaultFloatSimdFU(),
        MinorDefaultPredFU(),
        MinorDefaultMemFU(),
        MinorDefaultMiscFU(),
        # MinorDefaultVecFU(),
        # MinorDefaultVecFU(),
        ]

class RiscvMinorV2CPU(RiscvMinorCPU):
    executeFuncUnits = MinorV2FUPool()

class MinorV4FUPool(MinorFUPool):
    funcUnits = [
        MinorDefaultIntFU(),
        MinorDefaultIntFU(),
        MinorDefaultIntMulFU(),
        MinorDefaultIntDivFU(),
        MinorDefaultFloatSimdFU(),
        MinorDefaultPredFU(),
        MinorDefaultMemFU(),
        MinorDefaultMiscFU(),
        # MinorDefaultVecFU(),
        # MinorDefaultVecFU(),
        # MinorDefaultVecFU(),
        # MinorDefaultVecFU(),
        ]

class RiscvMinorV4CPU(RiscvMinorCPU):
    executeFuncUnits = MinorV4FUPool()
    executeCommitLimit = 4
    executeMemoryCommitLimit = 1

class L1Cache(Cache):
    """Simple L1 Cache with default values"""

    assoc = 8
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 16
    tgts_per_mshr = 20

    def connectBus(self, bus):
        """Connect this cache to a memory-side bus"""
        self.mem_side = bus.cpu_side_ports

    def connectCPU(self, cpu):
        """Connect this cache's port to a CPU-side port
        This must be defined in a subclass"""
        raise NotImplementedError

class L1ICache(L1Cache):
    """Simple L1 instruction cache with default values"""

    # Set the default size
    size = "8192kB" # is it enough for infinite ICache?

    def connectCPU(self, cpu):
        """Connect this cache's port to a CPU icache port"""
        self.cpu_side = cpu.icache_port

valid_cpu = {
    "X86AtomicSimpleCPU": X86AtomicSimpleCPU,
    "X86TimingSimpleCPU": X86TimingSimpleCPU,
    "X86DerivO3CPU": X86O3CPU,
    "ArmAtomicSimpleCPU": ArmAtomicSimpleCPU,
    "ArmTimingSimpleCPU": ArmTimingSimpleCPU,
    "ArmMinorCPU": ArmMinorCPU,
    "ArmDerivO3CPU": ArmO3CPU,
    "RiscvAtomicSimpleCPU": RiscvAtomicSimpleCPU,
    "RiscvTimingSimpleCPU": RiscvTimingSimpleCPU,
    "RiscvMinorCPU": RiscvMinorCPU,
    "RiscvDerivO3CPU": RiscvO3CPU,
    "RiscvMinorCPU": RiscvMinorCPU,
    "RiscvCustomCPU": RiscvCustomCPU,
    "RiscvMinorV2CPU": RiscvMinorV2CPU,
    "RiscvMinorV4CPU": RiscvMinorV4CPU,
    "RiscvVPU": RiscvVPU,
    "RiscvSparseVPU": RiscvSparseVPU,
}

valid_mem = {"SimpleMemory": MySimpleMemory, "ScratchpadMemory": SpadMemory, "DDR3_1600_8x8": DDR3_1600_8x8}

#parser = argparse.ArgumentParser()
#parser.add_argument("binary", type=str)
#parser.add_argument("--cpu", choices=valid_cpu.keys(), default="RiscvTimingSimpleCPU")
parser.add_argument("--cpu", choices=valid_cpu.keys(), default="RiscvMinorCPU")
parser.add_argument("--mem", choices=valid_mem.keys(), default="ScratchpadMemory")
parser.add_argument("--sparse", type=bool, default=False)
parser.add_argument("--vlane", type=int, default=128)

args = parser.parse_args()

# change systolicArrayWidth and systolicArrayHeight into args.vlane
SystolicArray.systolicArrayWidth = args.vlane
SystolicArray.systolicArrayHeight = args.vlane

system = System()

thispath = os.path.dirname(os.path.realpath(__file__))
binary = args.cmd
#binary = os.path.join(
#        thispath,
#        "../../../",
#        args.binary,
#)

#system.workload = SEWorkload.init_compatible(args.binary)
system.workload = SEWorkload.init_compatible(binary)

system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "1GHz"
system.clk_domain.voltage_domain = VoltageDomain()

if args.cpu not in (
    "X86AtomicSimpleCPU",
    "ArmAtomicSimpleCPU",
    "RiscvAtomicSimpleCPU",
):
    system.mem_mode = "timing"

system.mem_ranges = [AddrRange("512MB")]

system.cpu = valid_cpu[args.cpu]()

system.membus = SpmXBar(
    width = 64,
    frontend_latency = 0,
    forward_latency = 0,
    response_latency = 0)
# system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports

system.cpu.l1i = L1ICache()
system.cpu.l1i.connectCPU(system.cpu)
system.cpu.l1i.connectBus(system.membus)

system.cpu.createInterruptController()
if args.cpu in ("X86AtomicSimpleCPU", "X86TimingSimpleCPU", "X86DerivO3CPU"):
    system.cpu.interrupts[0].pio = system.membus.mem_side_ports
    system.cpu.interrupts[0].int_master = system.membus.cpu_side_ports
    system.cpu.interrupts[0].int_slave = system.membus.mem_side_ports

system.mem_ctrl = valid_mem[args.mem]()
system.mem_ctrl.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports
system.system_port = system.membus.cpu_side_ports

process = Process()
#process.cmd = [args.binary]
process.cmd = [binary] + args.options.split()
system.cpu.workload = process
system.cpu.createThreads()

root = Root(full_system=False, system=system)
m5.instantiate()

exit_event = m5.simulate()

if exit_event.getCause() != "exiting with last active thread context":
    exit(1)

# print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")
print(f"{m5.curTick() / 1000}")
print(f"{m5.curTick()}")

