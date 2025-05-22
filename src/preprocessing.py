import torch
from abc import ABC, abstractmethod


class PreprocessingModule(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def __call__(self, *args, reverse=False, **kwargs):
        if reverse:
            return self._backward(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    @abstractmethod
    def _forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def _backward(self, *args, **kwargs):
        pass


class MaskUnfoldedXY(PreprocessingModule):
    def __init__(self, mask_unfolded):
        self.mask_unfolded = mask_unfolded


    def _forward(self, x, y):
        assert x.shape[:-1] == self.mask_unfolded.shape
        assert y.shape == self.mask_unfolded.shape

        return x[self.mask_unfolded], y[self.mask_unfolded]


    def _backward(self, x, y):
        new_x_shape = self.mask_unfolded.shape + (x.shape[-1],)
        new_y_shape = self.mask_unfolded.shape 

        x_unmasked = torch.zeros(new_x_shape, device=x.device)
        x_unmasked[self.mask_unfolded] = x

        y_unmasked = torch.zeros(new_y_shape, device=y.device)
        y_unmasked[self.mask_unfolded] = y

        return x_unmasked, y_unmasked

