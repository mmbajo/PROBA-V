from typing import List, Tuple, Dict
from parseConfig import parseConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MarkNet(nn.Module):
    def __init__(self, cfg):
        super(MarkNet, self).__init__()

        self.moduleDefinitions =
