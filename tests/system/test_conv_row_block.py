"""BLOCKED: the conv template's X load cannot be made affine by a block choice. Its pixel index is nhw // OUT_W
and nhw % OUT_W, which PtrAnalysis structures only when OUT_W divides the program's base offset pid * BLOCK_M;
no power of two is a multiple of a YOLO output width, and Triton's frontend takes no other arange range."""
import os
import sys

import torch
sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from result_check import check_result

#: The output widths of yolov5n, yolov5s and yolov8n at 640 px, stem first.
YOLO_OUT_W = (320, 160, 80, 40, 20, 10)


def test_no_power_of_two_block_holds_a_whole_yolo_output_row():
    """Every YOLO width carries a factor of five, so no BLOCK_M Triton accepts is a multiple of one: the
    phase of pid * BLOCK_M against OUT_W varies with the program and the access stays a gather."""
    aligned = [w for w in YOLO_OUT_W if any((1 << i) % w == 0 for i in range(21))]
    check_result("ConvRowBlockNoPowerOfTwo", torch.tensor(len(aligned)), torch.tensor(0))


def test_triton_refuses_a_block_that_is_not_a_power_of_two():
    """The row-aligned block for the stem is 1280 (four rows of 320) and it reaches the compiler: measured
    2026-10-31, the conv heuristic emitted it and triton-compile refused it here. Flips when the pin moves."""
    from triton.language.semantic import TritonSemantic
    semantic = TritonSemantic.__new__(TritonSemantic)
    said = ""
    try:
        semantic.arange(0, 1280)
    except ValueError as e:
        said = str(e)
    except AttributeError:
        said = "the range was taken; the call went on to the builder this instance has none of"
    check_result("ConvRowBlockTritonRefuses", torch.tensor([bool("power of 2" in said)]),
                 torch.tensor([True]))


if __name__ == "__main__":
    test_no_power_of_two_block_holds_a_whole_yolo_output_row()
    test_triton_refuses_a_block_that_is_not_a_power_of_two()
