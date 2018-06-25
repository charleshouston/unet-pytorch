import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd._functions.utils import prepare_onnx_paddings
from torch.nn.modules.utils import _ntuple

from typing import Union, Tuple


def flip(x: Tensor, dim: int) -> Tensor:
    """Flip torch Tensor along given dimension axis."""
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous().view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:,
            getattr(torch.arange(x.size(1)-1, -1, -1),
                    ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class ReflectionPad3d(nn.Module):
    """Wrapper for ReflectionPadNd function in 3 dimensions."""
    def __init__(self, padding: Union[int, Tuple[int]]):
        super(ReflectionPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)

    def forward(self, x: Tensor) -> Tensor:
        return ReflectionPadNd.apply(x, self.padding)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ReflectionPadNd(Function):
    """Padding for same convolutional layer."""

    @staticmethod
    def symbolic(g, x: Tensor, padding: Union[int, Tuple[int]]):
        paddings = prepare_onnx_paddings(len(x.type().sizes()), pad)
        return g.op("Pad", x, pads_i=paddings, mode_s="reflect")

    @staticmethod
    def forward(ctx: Function, x: Tensor, pad: Tuple[int]) -> Tensor:
        ctx.pad = pad
        ctx.input_size = x.size()
        ctx.l_inp = len(x.size())
        ctx.pad_tup = tuple([(a, b)
                             for a, b in zip(pad[:-1:2], pad[1::2])]
                            [::-1])
        ctx.l_pad = len(ctx.pad_tup)
        ctx.l_diff = ctx.l_inp - ctx.l_pad
        assert ctx.l_inp >= ctx.l_pad

        new_dim = tuple([sum((d,) + ctx.pad_tup[i])
                         for i, d in enumerate(x.size()[-ctx.l_pad:])])
        assert all([d > 0 for d in new_dim]), 'input is too small'

        # Create output tensor by concatenating with reflected chunks.
        output = x.new(x.size()[:(ctx.l_diff)] + new_dim).zero_()
        c_input = x

        for i, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] > 0:
                chunk1 = flip(c_input.narrow(i, 0, pad[0]), i)
                c_input = torch.cat((chunk1, c_input), i)
            if p[1] > 0:
                chunk2 = flip(c_input.narrow(i, c_input.shape[i]-p[1], p[1]), i)
                c_input = torch.cat((c_input, chunk2), i)
        output.copy_(c_input)
        return output

    @staticmethod
    def backward(ctx: Function, grad_output: Tensor) -> Tensor:
        grad_input = Tensor(grad_output.data.new(ctx.input_size).zero_())
        grad_input_slices = [slice(0, x,) for x in ctx.input_size]

        cg_output = grad_output
        for i_s, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] > 0:
                cg_output = cg_output.narrow(i_s, p[0],
                                             cg_output.size(i_s) - p[0])
            if p[1] > 0:
                cg_output = cg_output.narrow(i_s, 0,
                                             cg_output.size(i_s) - p[1])
        gis = tuple(grad_input_slices)
        grad_input[gis] = cg_output

        return grad_input, None, None


class Softmax3d(nn.Module):
    """Applies softmax over features for each spatial location.

    Expects a volumetric image of dimensions `(N, C, D, H, W)`.
    """

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 5, 'Softmax3d requires a 5D Tensor.'
        return F.softmax(x, 1, _stacklevel=5)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

