from typing import List, Tuple, Dict
from parseConfig import parseConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class MarkNet(nn.Module):
    # We shall complete this class after the experiments

    def __init__(self, cfg):
        super(MarkNet, self).__init__()

        self.moduleDefinitions = parseConfig(cfg)
        pass


class Conv3DResNet(nn.Module):
    def __init__(self, inputSize=(1, 9, 32, 32), upSampleScale: int, numFilters: int):
        super(Conv3DResNet, self).__init__()

        numLowResImg, C, H, W = inputSize
        padding = samePaddingForConv3d(inputSize, (3, 3, 3), (1, 1, 1))
        # Same Padding Mode
        self.conv3dOne = weightedNormConv3d(inChannels=C, outChannels=32, kernelSize=(3, 3, 3),
                                            stride=(1, 1, 1), padding=padding, paddingMode='zeros')
        # Valid Padding Mode
        self.conv3dTwo = weightedNormConv3d(inChannels=C, outChannels=32, kernelSize=(3, 3, 3),
                                            stride=(1, 1, 1), padding=padding, paddingMode='reflect')
        # Valid Padding Mode
        self.conv3dThree = weightedNormConv3d(inChannels=C, outChannels=32, kernelSize=(3, 3, 3),
                                              stride=(1, 1, 1), padding=padding, paddingMode='reflect')

    def forward(self, x: torch.Tensor):
        x = self.conv3dFromInput(x)
        x = F.relu(x)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x.mul_(F.softplus(x).tanh())


def ResConv3d(xIn: torch.Tensor, conv3dNet: nn.Module, numFilters: int,
              kernelSize: tuple, expRate: int, decayRate: float) -> torch.Tensor:
    n, c, d, h, w = xIn.shape
    # Expand Block | Same padding
    padding = samePaddingForConv3d(xIn.shape, (1, 1, 1), (1, 1, 1))
    x = conv3dNet(inChannels=c, outChannels=numFilters*expRate, kernelSize=(1, 1, 1),
                  stride=(1, 1, 1), padding=padding, paddingMode='zeros')(xIn)
    x = F.relu(x)
    # Decay Block | Same padding
    padding = samePaddingForConv3d(x.shape, (1, 1, 1), (1, 1, 1))
    x = conv3dNet(inChannels=numFilters*expRate, outChannels=int(numFilters*decayRate), kernelSize=(1, 1, 1),
                  stride=(1, 1, 1), padding=padding, paddingMode='zeros')(x)
    # Norm Block | Same padding
    padding = samePaddingForConv3d(x.shape, kernelSize, (1, 1, 1))
    x = conv3dNet(inChannels=int(numFilters*decayRate), outChannels=numFilters, kernelSize=kernelSize,
                  stride=(1, 1, 1), padding=padding, paddingMode='zeros')(x)
    return xIn + x


def ReflectConv3d(xIn: torch.Tensor, conv3dNet: nn.Module, kernelSize: Tuple, depth: int, scale: int) -> torch.Tensor:
    n, c, d, h, w = xIn.shape
    padding = samePaddingForConv3d(xIn.shape, kernelSize, (1, 1, 1))
    # Reflect | Same padding
    x = conv3dNet(inChannels=c, outChannels=numFilters*expRate, kernelSize=kernelSize,
                  stride=(1, 1, 1), padding=padding, paddingMode='reflect')(xIn)
    for _ in range(1, depth//scale):
        padding = samePaddingForConv3d(x.shape, kernelSize, (1, 1, 1))
        x = conv3dNet(inChannels=c, outChannels=numFilters*expRate, kernelSize=kernelSize,
                      stride=(1, 1, 1), padding=padding, paddingMode='reflect')(x)
    return x


def UpSampleConv3d(xIn: torch.Tensor,
                   conv3dNet: nn.Module,
                   upSampleScale: int, kernelSize: tuple) -> torch.Tensor:
    # Valid padding
    n, c, d, h, w = xIn.shape
    x = conv3dNet(inChannels=c, outChannels=upSampleScale**2, kernelSize=kernelSize,
                  stride=(1, 1, 1), padding=(0, 0, 0), paddingMode='zeros')(xIn)
    return x


def ReshapeToOutput(xIn: torch.Tensor, targetOutputShape: tuple):
    x = Reshape(targetOutputShape)(xIn)
    x = DepthToSpace(3)(x)
    return x


def weightedNormConv3d(inChannels: int, outChannels: int, kernelSize: Tuple[int],
                       stride: Tuple[int], padding: Tuple[int], paddingMode='zeros') -> nn.Module:
    return nn.utils.weight_norm(nn.Conv3d(inChannels, outChannels, kernelSize,
                                          stride, padding, paddingMode))


def samePaddingForConv3d(inputSize: Tuple[int], kernelSize: Tuple[int], stride: Tuple[int]) -> Tuple[int]:
    _, dIn, hIn, wIn = inputSize

    dPad = (((dIn - 1) * stride[0]) - dIn + (kernelSize[0] - 1) + 1) // 2
    hPad = (((hIn - 1) * stride[1]) - hIn + (kernelSize[1] - 1) + 1) // 2
    wPad = (((wIn - 1) * stride[2]) - wIn + (kernelSize[2] - 1) + 1) // 2

    padding = (dPad, hPad, wPad)
    return padding
