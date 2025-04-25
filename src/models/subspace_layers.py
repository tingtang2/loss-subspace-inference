# implementation from https://github.com/vaseline555/SuPerFed/blob/main/src/models/layers.py

import torch
from torch import nn
from torch.nn import functional as F


# Linear layer implementation
class SubspaceLinear(nn.Linear):

    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w = self.get_weight()
        x = F.linear(input=x, weight=w, bias=self.bias)
        return x


# TODO: add bias weight and retrain
class TwoParamLinear(SubspaceLinear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_1 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, seed):
        if seed == -1:  # SCAFFOLD
            torch.nn.init.zeros_(self.weight_1)
        else:
            torch.manual_seed(seed)
            torch.nn.init.xavier_normal_(self.weight_1)


class LinesLinear(TwoParamLinear):

    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight_1
        return w


# TODO: add bias weight and retrain
class ThreeParamLinear(SubspaceLinear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_1 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight_2 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, seed):
        if seed == -1:  # SCAFFOLD
            torch.nn.init.zeros_(self.weight_1)
            torch.nn.init.zeros_(self.weight_2)
        else:
            torch.manual_seed(seed)
            torch.nn.init.xavier_normal_(self.weight_1)
            torch.nn.init.xavier_normal_(self.weight_2)


class SimplexLinear(ThreeParamLinear):

    def get_weight(self):
        mult = 1 - self.t1 - self.t2
        w = mult * self.weight + self.t1 * self.weight_1 + self.t2 * self.weight_2
        return w
