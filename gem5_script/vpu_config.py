import os
import m5
from m5.objects import *

#: One FU whose latency comes from the tile, instead of four latency buckets.
#: Off by default -- see CrossLaneUnit below.
_XLU_FU = os.environ.get("TORCHSIM_COMPILE_XLU_FU", "0") == "1"

class SystolicArray(MinorFU):
    unitType = "SystolicArray"
    opClasses = minorMakeOpClassSet(["CustomMatMul", "CustomMatMuliVpush", "CustomMatMulwVpush", "CustomMatMulvpop"])
    opLat = 1
    systolicArrayWidth = 128
    systolicArrayHeight = 128

class SparseAccelerator(MinorFU):
    unitType = "SparseAccelerator"
    opClasses = minorMakeOpClassSet(["CustomMatMul", "CustomMatMuliVpush", "CustomMatMulwVpush", "CustomMatMulvpop"])
    opLat = 1

# THERE IS ONE CROSS-LANE UNIT AND FOUR FUs, because a MinorFU is a cost and not
# a machine. The unit holds one tile in one queue pair; what differs between its
# operations is the SIMM5 XU field -- whether a pass swaps depth and lane -- and
# that is a factor of two in the serialiser. The decoder routes each arm to the
# class whose number matches, so these names read as costs: "Transpose" is any
# pass that crosses (transpose, all-gather), "Crossbar" any that only shuffles
# lanes (broadcast, permute). Splitting costs no fidelity here because issueLat
# is 1 on all four, so neither spelling models the unit's occupancy.
class TransposeUnit(MinorFU):
    # A pass that swaps depth and lane. opLat IS THE SERIALISER: the unit takes
    # (m+n)-1 passes over a m x n tile and one push carries vlen/32 = 16 values
    # per lane, so a square tile of depth D costs ~2D passes over D/16 pushes --
    # 32 per push, and the ratio holds at every square size. The pop only drains,
    # so it costs one.
    # ALL-GATHER IS 32 TOO, AND THAT IS THE NUMBER RATHER THAN A PLACEHOLDER.
    # opLat here counts the SERIALISER and nothing else -- the crossbar's own 16
    # already discards a tree 8 stages deep on the grounds that it rides the same
    # push -- and a combination is ONE pass over ONE serialiser. The post-RPU can
    # start as soon as the crossing has produced row 0, so it streams behind the
    # crossing instead of queueing after it. What the old two-pass spelling paid
    # twice was the serialiser itself, plus a round trip through a vector register
    # and dst_bank that no opLat here ever modelled.
    opClasses = minorMakeOpClassSet(["CustomTransposePush"])
    opLat = 32

class TransposePopUnit(MinorFU):
    opClasses = minorMakeOpClassSet(["CustomTransposePop"])
    opLat = 1

class CrossbarUnit(MinorFU):
    # A pass that shuffles lanes without crossing: replicate one lane to all, or
    # read the lane each lane names. opLat IS THE SERIALISER, as above, and it is
    # the whole cost here: a crossbar is one stage deep but pipelined behind a
    # push that carries vlen/32 = 16 values per lane, so 16 depth slices cost 16.
    # The replicate shares the class because it shares that serialiser, which is
    # what dominates -- a fan-out is not cheaper than the wire it goes down.
    opClasses = minorMakeOpClassSet(["CustomCrossbarPush"])
    opLat = 16

class CrossbarPopUnit(MinorFU):
    opClasses = minorMakeOpClassSet(["CustomCrossbarPop"])
    opLat = 1

#: THE UNIT AS ONE FU, WITH THE PASS'S COST COMING FROM THE TILE. The four FUs
#: below are four *latency buckets*, not four machines: a `MinorFU` carries one
#: `opLat` and the four costs differ, so they had to be split. `CrossLaneFU`
#: (gem5 `func_unit.hh`) holds the pass's state instead -- `depth` counts what the
#: pushes handed over and the first pop fires the pass -- so one FU says
#: `m + n - 1` for a crossing and `2m + n` for a lane-only shuffle, at any shape.
#: SystolicArrayFU already does exactly this for the array; this is that pattern
#: a second time.
#:
#: BEHIND A FLAG because the four-FU spelling is what every other run has been
#: measured against. `TORCHSIM_COMPILE_XLU_FU=1` selects this one.
class CrossLaneUnit(MinorFU):
    unitType = "CrossLane"
    crossLaneWidth = int(os.environ.get("TORCHSIM_COMPILE_XLU_LANES", "256"))
    opClasses = minorMakeOpClassSet(["CustomTransposePush", "CustomTransposePop",
                                     "CustomCrossbarPush",  "CustomCrossbarPop"])
    opLat = 1          # issue only; the pass's cost is the FU's own
    issueLat = 1

class SpecialFunctionUnit(MinorFU):
    opClasses = minorMakeOpClassSet([
        "CustomVexp",
        "CustomVerf",
        "CustomVtanh",
        "CustomVsin",
        "CustomVcos",
        "CustomVlog",
        "CustomVatan",
        ])
    opLat = 10

class MinorFPUnit(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "FloatAdd",
            "FloatCmp",
            "FloatCvt",
            "FloatMult",
            "FloatMultAcc",
            "FloatDiv",
            "FloatMisc",
            "FloatSqrt"
        ]
    )

class MinorVecAdder(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdAdd",
            "SimdFloatAdd",
            "SimdFloatAlu",
            "SimdFloatCmp",
            "SimdShift",
            "SimdShiftAcc",
            "SimdAddAcc",
            "SimdAlu",
            "SimdCmp",
        ]
    )
    opLat = 1

class MinorVecMultiplier(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdMult",
            "SimdFloatMult",
            "SimdMultAcc",
            "SimdMatMultAcc",
            "SimdSqrt",
            "SimdFloatMultAcc",
            "SimdFloatMatMultAcc",
            "SimdFloatSqrt",
        ]
    )
    opLat = 1

class MinorVecDivider(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdDiv",
            "SimdFloatDiv",
        ]
    )
    opLat = 1

class MinorVecReduce(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdReduceAdd",
            "SimdReduceAlu",
            "SimdReduceCmp",
            "SimdFloatReduceAdd",
            "SimdFloatReduceCmp",
        ]
    )
    opLat = 1

class MinorVecLdStore(MinorFU):
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
            "SimdUnitStrideSegmentedLoad",
            "SimdUnitStrideSegmentedStore",
        ]
    )
    opLat = 1

class MinorVecMisc(MinorFU):
    opClasses = minorMakeOpClassSet(
        [
            "SimdCvt",
            "SimdFloatCvt",
            "SimdFloatMisc",
            "SimdPredAlu",
            "SimdMisc",
            "SimdExt",
            "SimdFloatExt",
            "CustomVlaneIdx",
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

class MinorCustomIntFU(MinorDefaultIntFU):
    opLat = 1

class MinorCustomIntDivFU(MinorDefaultIntDivFU):
    opLat = 1

class MinorCustomIntMulFU(MinorDefaultIntMulFU):
    opLat = 1

class MinorCustomPredFU(MinorDefaultPredFU):
    opLat = 1

class MinorCustomMemFU(MinorDefaultMemFU):
    opLat = 1

class MinorCustomMiscFU(MinorDefaultMiscFU):
    opLat = 1

class MinorCustomFUPool(MinorFUPool):
    funcUnits = [
        # Scalar unit
        MinorFPUnit(),
        MinorCustomIntFU(),
        MinorCustomIntFU(),
        MinorCustomIntMulFU(),
        MinorCustomIntDivFU(),
        MinorCustomPredFU(),
        MinorCustomMemFU(),
        MinorCustomMiscFU(),

        # Scalar unit
        MinorFPUnit(),
        MinorCustomIntFU(),
        MinorCustomIntFU(),
        MinorCustomIntMulFU(),
        MinorCustomIntDivFU(),
        MinorCustomPredFU(),
        MinorCustomMemFU(),
        MinorCustomMiscFU(),

        # Matmul unit
        SystolicArray(), # 0
 
        # Vector
        MinorVecConfig(), # 1 for vector config
        MinorVecConfig(),
        MinorVecMisc(),
        MinorVecMisc(),
        MinorVecLdStore(),
        MinorVecLdStore(),

        # Vector ALU0
        MinorVecAdder(), # 6
        MinorVecMultiplier(), # 7
        MinorVecDivider(), # 8
        MinorVecReduce(),

        # Vector ALU1
        MinorVecAdder(), # 18 ~ 29
        MinorVecMultiplier(),
        MinorVecDivider(),
        MinorVecReduce(),

        # Vector
        MinorVecConfig(), # 1 for vector config
        MinorVecConfig(),
        MinorVecMisc(),
        MinorVecMisc(),
        MinorVecLdStore(),
        MinorVecLdStore(),

        # Vector ALU0
        MinorVecAdder(), # 6
        MinorVecMultiplier(), # 7
        MinorVecDivider(), # 8
        MinorVecReduce(),

        # Vector ALU1
        MinorVecAdder(), # 18 ~ 29
        MinorVecMultiplier(),
        MinorVecDivider(),
        MinorVecReduce(),

        # SFU
        SpecialFunctionUnit(),

        # Cross-lane
    ] + ([CrossLaneUnit()] if _XLU_FU else
         [TransposeUnit(), TransposePopUnit(), CrossbarUnit(), CrossbarPopUnit()])

class RiscvVPU(RiscvMinorCPU):
    fetch1FetchLimit = 8
    decodeInputWidth = 8
    fetch1ToFetch2BackwardDelay = 0
    fetch2InputBufferSize = 8
    decodeInputBufferSize = 8
    decodeInputWidth = 8
    executeInputBufferSize = 128
    executeInputWidth = 12
    executeIssueLimit = 12
    executeCommitLimit = 12

    # Memory
    executeMemoryIssueLimit = 8
    executeMemoryCommitLimit = 8
    executeMaxAccessesInMemory = 8
    executeLSQMaxStoreBufferStoresPerCycle = 8
    executeLSQTransfersQueueSize = 8
    executeLSQStoreBufferSize = 8

    executeFuncUnits = MinorCustomFUPool()
