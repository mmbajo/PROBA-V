from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Conv3DResNet(nn.Module):
    def __init__(self, inputSize: tuple, upSampleScale: int, numResBlocks: int,
                 kernelSize: tuple, numFilters: int, expRate: int, decayRate: float):
        super().__init__()
        self.numResBlocks = numResBlocks
        # Layers
        self.NormConv3D = NormConv3D(inputSize, numFilters, kernelSize, 'zeros')  # (1, 9, 34, 34)
        self.ResBlockConv3D = ResBlockConv3D(numFilters, kernelSize, expRate, decayRate)
        self.UpSampleConv3D = UpSampleConv3D(numFilters, numFilters, kernelSize, upSampleScale, inputSize[-1] - 2)
        self.Reshape = Reshape((inputSize[-1] - 2, inputSize[-1] - 2, 9))
        self.DepthToSpace = DepthToSpace(3)

    def forward(self, x: torch.Tensor):
        x = self.NormConv3D(x)
        x = F.relu(x)

        # Residual Blocks Section
        for _ in range(self.numResBlocks):
            x = self.ResBlockConv3D(x)

        # Upsample Block
        x = self.UpSampleConv3D(x)

        x = self.Reshape(x)
        x = self.DepthToSpace(x)

        return x


class ResBlockConv3D(nn.Module):
    def __init__(self, numFilters: int, kernelSize: tuple,
                 expRate: int, decayRate: float):
        super(ResBlockConv3D, self).__init__()
        self.numFilters = numFilters
        self.kernelSize = kernelSize
        self.expRate = expRate
        self.decayRate = decayRate

    def forward(self, xIn):
        x = ExpConv3D(xIn.shape, self.numFilters, (1, 1, 1), self.expRate)(xIn)
        x = DecConv3D(x.shape, self.numFilters, (1, 1, 1), self.decayRate)(x)
        x = NormConv3D(x.shape, self.numFilters, self.kernelSize, 'zeros')(x)
        return x + xIn


class ExpConv3D(nn.Module):
    def __init__(self, inputShape: tuple, numFilters: int, kernelSize: tuple, expRate: int):
        super(ExpConv3D, self).__init__()
        padding = samePaddingForConv3d(inputShape, kernelSize, (1, 1, 1))
        self.expandBlock = nn.utils.weight_norm(nn.Conv3d(in_channels=inputShape[1], out_channels=numFilters*expRate,
                                                          kernel_size=kernelSize, stride=(1, 1, 1), padding=padding,
                                                          padding_mode='zeros'),
                                                name='weight')

    def forward(self, x):
        x = self.expandBlock(x)
        x = F.relu(x)
        return x


class DecConv3D(nn.Module):
    def __init__(self, inputShape: tuple, numFilters: int, kernelSize: tuple, decayRate: float):
        super(DecConv3D, self).__init__()
        padding = samePaddingForConv3d(inputShape, kernelSize, (1, 1, 1))
        self.decayBlock = nn.utils.weight_norm(nn.Conv3d(in_channels=inputShape[1], out_channels=numFilters*decayRate,
                                                         kernel_size=kernelSize, stride=(1, 1, 1), padding=padding,
                                                         padding_mode='zeros'),
                                               name='weight')

    def forward(self, x):
        x = self.decayBlock(x)
        return x


class NormConv3D(nn.Module):
    def __init__(self, inputShape: tuple, numFilters: int, kernelSize: tuple, paddingMode: str):
        super(NormConv3D, self).__init__()
        padding = samePaddingForConv3d(inputShape, kernelSize, (1, 1, 1))
        self.normBlock = nn.utils.weight_norm(nn.Conv3d(in_channels=inputShape[1], out_channels=numFilters,
                                                        kernel_size=kernelSize, stride=(1, 1, 1), padding=padding,
                                                        padding_mode=paddingMode),
                                              name='weight')

    def forward(self, x):
        x = self.normBlock(x)
        return x


class UpSampleConv3D(nn.Module):
    def __init__(self, inputShape: tuple, numFilters: int, kernelSize: tuple, upSampleScale: int, initialDepth: int):
        super(UpSampleConv3D, self).__init__()
        self.numFilters = numFilters
        self.initialDepth = initialDepth
        self.upSampleBlock = nn.utils.weight_norm(nn.Conv3d(in_channels=inputShape, out_channels=upSampleScale**2,
                                                            kernel_size=kernelSize, stride=(1, 1, 1),
                                                            padding_mode='zeros'),
                                                  name='weight')

    def forward(self, x):
        x = NormConv3D(x.shape, numFilters, kernelSize, 'reflect')(x)
        for _ in range(1, self.initialDepth//scale):
            x = NormConv3D(x.shape, numFilters, kernelSize, 'reflect')(x)
        x = self.upSampleBlock(x)
        return x


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
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


def samePaddingForConv3d(inputSize: Tuple[int], kernelSize: Tuple[int], stride: Tuple[int]) -> Tuple[int]:
    _, dIn, hIn, wIn = inputSize

    dPad = (((dIn - 1) * stride[0]) - dIn + (kernelSize[0] - 1) + 1) // 2
    hPad = (((hIn - 1) * stride[1]) - hIn + (kernelSize[1] - 1) + 1) // 2
    wPad = (((wIn - 1) * stride[2]) - wIn + (kernelSize[2] - 1) + 1) // 2

    padding = (dPad, hPad, wPad)
    return padding
