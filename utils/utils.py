from typing import List, Tuple


def samePaddingForConv3d(inputSize: Tuple[int], kernelSize: Tuple[int], stride: Tuple[int]) -> Tuple[int]:
    _, dIn, hIn, wIn = inputSize

    dPad = (((dIn - 1) * stride[0]) - dIn + (kernelSize[0] - 1) + 1) // 2
    hPad = (((hIn - 1) * stride[1]) - hIn + (kernelSize[1] - 1) + 1) // 2
    wPad = (((wIn - 1) * stride[2]) - wIn + (kernelSize[2] - 1) + 1) // 2

    padding = (dPad, hPad, wPad)
    return padding
