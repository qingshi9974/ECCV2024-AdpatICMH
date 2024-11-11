import torch
import torch.nn.functional as F
from numpy import ceil


def cat_k(input):
    """concat second dimesion to batch"""
    return input.flatten(0, 1)


def split_k(input, size: int, dim: int = 0):
    """reshape input to original batch size"""
    if dim < 0:
        dim = input.dim() + dim
    split_size = list(input.size())
    split_size[dim] = size
    split_size.insert(dim+1, -1)
    return input.view(split_size)


class Alignment(torch.nn.Module):
    """Image Alignment for model downsample requirement"""

    def __init__(self, divisor=64., mode='pad', padding_mode='replicate'):
        super().__init__()
        self.divisor = float(divisor)
        self.mode = mode
        self.padding_mode = padding_mode
        self._tmp_shape = None
        self.value = 0

    def extra_repr(self):
        s = 'divisor={divisor}, mode={mode}'
        if self.mode == 'pad':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    @staticmethod
    def _resize(input, size):
        return F.interpolate(input, size, mode='bilinear', align_corners=False)

    def _align(self, input):
        H, W = input.size()[-2:]
        H_ = int(ceil(H / self.divisor) * self.divisor)
        W_ = int(ceil(W / self.divisor) * self.divisor)
        pad_H, pad_W = H_-H, W_-W
        if pad_H == pad_W == 0:
            self._tmp_shape = None
            return input

        self._tmp_shape = input.size()
        if self.mode == 'pad':
            if self.padding_mode =='constant':
                return F.pad(input, (0, pad_W, 0, pad_H), mode=self.padding_mode,value=0)
            else:
                return F.pad(input, (0, pad_W, 0, pad_H), mode=self.padding_mode)
        elif self.mode == 'resize':
            return self._resize(input, size=(H_, W_))

    def _resume(self, input, shape=None):
        if shape is not None:
            self._tmp_shape = shape
        if self._tmp_shape is None:
            return input

        if self.mode == 'pad':
            output = input[..., :self._tmp_shape[-2], :self._tmp_shape[-1]]
        elif self.mode == 'resize':
            output = self._resize(input, size=self._tmp_shape[-2:])

        return output

    def align(self, input):
        """align"""
        if input.dim() == 4:
            return self._align(input)
        elif input.dim() == 5:
            return split_k(self._align(cat_k(input)), input.size(0))

    def resume(self, input, shape=None):
        """resume"""
        if input.dim() == 4:
            return self._resume(input, shape)
        elif input.dim() == 5:
            return split_k(self._resume(cat_k(input), shape), input.size(0))

    def forward(self, func, *args, **kwargs):
        pass