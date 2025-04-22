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


# Nonlinear layer implementation


class SubspaceNonLinear(nn.Linear):

    def forward(self, x):
        w = self.get_weight().reshape(self.weight.size())
        x = F.linear(input=x, weight=w, bias=self.bias)
        return x


# Nonlinear 1D subspace #
class TwoParamNonLinear(SubspaceNonLinear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.line = ParameterizedSubspace(n_in=1,
                                          n_out=torch.numel(self.weight))


class LinesNN(TwoParamNonLinear):

    def get_weight(self):
        w = self.line.forward(self.alpha)
        return w


# Neural net parameterization for the loss subspace of arbitrary dimension
class ParameterizedSubspace(nn.Module):

    def __init__(self, n_in, n_out):
        super().__init__()
        self.parameterization_linear_1 = nn.Linear(n_in, 10)
        self.parameterization_linear_2 = nn.Linear(10, 20)
        self.parameterization_linear_3 = nn.Linear(20, 40)
        self.parameterization_linear_4 = nn.Linear(40, n_out)

    def forward(self, x):
        x = self.parameterization_linear_1(x)
        x = F.tanh(x)
        x = self.parameterization_linear_2(x)
        x = F.tanh(x)
        x = self.parameterization_linear_3(x)
        x = F.tanh(x)
        x = self.parameterization_linear_4(x)
        return x
