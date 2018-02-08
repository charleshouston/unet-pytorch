import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Softmax3d(nn.Module):
    """Applies softmax over features for each spatial location.

    Expects a volumetric image of dimensions `(N, C, D, H, W)`.
    """

    def forward(self, input: Variable) -> Variable:
        assert input.dim() == 5, 'Softmax3d requires a 5D Tensor.'
        return F.softmax(input, 1, _stacklevel=5)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
