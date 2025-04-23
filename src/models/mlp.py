from models.subspace_layers import LinesLinear, LinesNN, SimplexLinear
from torch import nn
from torch.nn import functional as F

## Standard MLP ##


class MLP(nn.Module):

    def __init__(self, n_in, n_out, dropout_prob=0.15):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x


class NN(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim, dropout_prob) -> None:
        super().__init__()

        self.mlp = MLP(n_in=input_dim,
                       n_out=hidden_dim,
                       dropout_prob=dropout_prob)
        self.out = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        x = self.mlp(x)

        return self.out(x)


## Standard MLP ##

## Linear subspace ##


class SubspaceMLP(nn.Module):

    def __init__(self, n_in, n_out, seed, dropout_prob=0.15, num_weights=2):
        super().__init__()

        if num_weights == 2:
            self.linear = LinesLinear(n_in, n_out)
        elif num_weights == 3:
            self.linear = SimplexLinear(in_features=n_in, out_features=n_out)
        self.linear.initialize(seed)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x


class SubspaceNN(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 out_dim,
                 dropout_prob,
                 seed,
                 num_weights=2) -> None:
        super().__init__()

        self.mlp = SubspaceMLP(n_in=input_dim,
                               n_out=hidden_dim,
                               dropout_prob=dropout_prob,
                               seed=seed,
                               num_weights=num_weights)
        if num_weights == 2:
            self.out = LinesLinear(in_features=hidden_dim,
                                   out_features=out_dim)
        elif num_weights == 3:
            self.out = SimplexLinear(in_features=hidden_dim,
                                     out_features=out_dim)
        self.out.initialize(seed)

    def forward(self, x):
        x = self.mlp(x)

        return self.out(x)


## Linear subspace ##

## Non-Linear subspace ##


class NonLinearSubspaceMLP(nn.Module):

    def __init__(self, n_in, n_out, seed, dropout_prob=0.15):
        super().__init__()

        self.linear = LinesNN(n_in, n_out)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x


class NonLinearSubspaceNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim, dropout_prob,
                 seed) -> None:
        super().__init__()

        self.mlp = NonLinearSubspaceMLP(n_in=input_dim,
                                        n_out=hidden_dim,
                                        dropout_prob=dropout_prob,
                                        seed=seed)
        self.out = LinesNN(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        x = self.mlp(x)

        return self.out(x)


## Non-Linear subspace ##
