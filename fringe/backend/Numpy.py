from ..backend.core import CoreFunctions
import numpy as np


class Numpy(CoreFunctions):
    @staticmethod
    def convert(input_, dtype):
        return np.array(input_, dtype=dtype)

    @staticmethod
    def expand_dim(input_, axis):
        return np.expand_dims(input_, axis=axis)

    @staticmethod
    def repeat(input_, repeats, axis):
        return np.repeat(input_, repeats=repeats, axis=axis)

    @staticmethod
    def reduce_sum(input_, axis):
        return np.sum(input_, axis=axis)

    @staticmethod
    def pad(input_, padding, fill_value):
        return np.pad(input_, padding, mode="constant", constant_values=fill_value)

    @staticmethod
    def unpad(input_, padding):
        shape = input_.shape
        slc = [slice(None)] * input_.ndim
        for i, p in enumerate(padding):
            n = shape[i] - p[1]
            slc[i] = slice(p[0], n) if p[0] != 0 else slc[i]
        return input_[tuple(slc)]

    @staticmethod
    def abs(input_):
        return np.abs(input_)

    @staticmethod
    def angle(input_):
        return np.angle(input_)


    @staticmethod
    def fft(input_, dims):
        # x must be in the shape of 'NWHC' as 2D or 'NWC'/'NW' as 1D
        if dims == 1:
            return np.fft.fft(input_)
        elif dims == 2:
            return np.fft.fft2(input_)
        elif dims >= 3:
            return np.fft.fftn(input_)

    @staticmethod
    def ifft(input_, dims):
        # x must be in the shape of 'NWHC' as 2D or 'NWC'/'NW' as 1D
        if dims == 1:
            return np.fft.ifft(input_)
        elif dims == 2:
            return np.fft.ifft2(input_)
        elif dims == 3:
            return np.fft.ifftn(input_)

    @staticmethod
    def fftshift(input_):
        return np.fft.fftshift(input_)

    @staticmethod
    def exp(input_):
        return np.exp(input_)

    @staticmethod
    def divide(x, y):
        return x / y

    @staticmethod
    def sqrt(input_):
        return np.sqrt(input_)

    @staticmethod
    def multiply(x, y):
        return x * y

    @staticmethod
    def complex(real, imag):
        return real + 1j * imag

    @staticmethod
    def zeros_like(input_):
        return np.zeros_like(input_)

    @staticmethod
    def boolean_mask(input_, mask):
        return input_[mask]

    @staticmethod
    def logical_and(*args):
        return np.logical_and(*args)

    @staticmethod
    def meshgrid(x, y, indexing='xy'):
        return np.meshgrid(x, y, indexing=indexing)

    @staticmethod
    def linspace(start, stop, num):
        return np.linspace(start, stop, num)

    @staticmethod
    def where(condition, x, y):
        return np.where(condition, x, y)

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def zeros_like(a, dtype=np.float32):
        return np.zeros_like(a, dtype=dtype)
