from ..backend.core import CoreFunctions
import tensorflow as tf


class TensorFlow(CoreFunctions):
    @staticmethod
    def convert(input_, dtype):
        return tf.convert_to_tensor(input_, dtype=dtype)

    @staticmethod
    def expand_dim(input_, axis):
        return tf.expand_dims(input_, axis=axis)

    @staticmethod
    def repeat(input_, repeats, axis):
        return tf.repeat(input_, repeats=repeats, axis=axis)

    @staticmethod
    def reduce_sum(input_, axis):
        return tf.reduce_sum(input_, axis=axis)

    @staticmethod
    def pad(input_, padding, fill_value):
        return tf.pad(input_, padding, "CONSTANT", fill_value)

    @staticmethod
    def unpad(input_, padding):
        begin = []
        size = []
        shape = input_.shape
        # This loop converts a padding array like ((n11, n12), (n21, n22), ...) to
        # ([n11, n21, ...], [d - n11 - n12, d - n21 - n22, ...]) where d is length of x at each dimension.
        for i, n in enumerate(padding):
            n1 = n[0]
            n2 = n[1]
            d = shape[i]
            begin.append(n1)
            size.append(d - n1 - n2)
        return tf.slice(input_, begin, size)

    @staticmethod
    def abs(input_):
        return tf.math.abs(input_)

    @staticmethod
    def angle(input_):
        return tf.math.angle(input_)

    @staticmethod
    def fft(input_, dims):
        # x must be in the shape of 'NWHC' as 2D or 'NWC'/'NW' as 1D
        if dims == 1:
            return tf.signal.fft(input_)
        elif dims == 2:
            return tf.signal.fft2d(input_)
        elif dims == 3:
            return tf.signal.fft3d(input_)

    @staticmethod
    def ifft(input_, dims):
        # x must be in the shape of 'NWHC' as 2D or 'NWC'/'NW' as 1D
        if dims == 1:
            return tf.signal.ifft(input_)
        elif dims == 2:
            return tf.signal.ifft2d(input_)
        elif dims == 3:
            return tf.signal.ifft3d(input_)

    @staticmethod
    def fftshift(input_):
        return tf.signal.fftshift(input_)

    @staticmethod
    def exp(input_):
        return tf.math.exp(input_)

    @staticmethod
    def divide(x, y):
        return tf.math.divide(x, y)

    @staticmethod
    def sqrt(input_):
        return tf.math.sqrt(input_)

    @staticmethod
    def multiply(x, y):
        return tf.math.multiply(x, y)

    @staticmethod
    def complex(real, imag):
        return tf.complex(real, imag)

    @staticmethod
    def zeros_like(input_):
        return tf.zeros_like(input_)

    @staticmethod
    def boolean_mask(input_, mask):
        return tf.boolean_mask(input_, mask)

    @staticmethod
    def logical_and(*args):
        return tf.logical_and(*args)

    @staticmethod
    def meshgrid(x, y, indexing='xy'):
        return tf.meshgrid(x, y, indexing=indexing)

    @staticmethod
    def linspace(start, stop, num):
        return tf.linspace(start, stop, num)

    @staticmethod
    def where(condition, x, y):
        return tf.where(condition, x, y)

    @staticmethod
    def zeros(shape, dtype=tf.float32):
        return tf.zeros(shape, dtype=dtype)

    @staticmethod
    def zeros_like(a, dtype=tf.float32):
        return tf.zeros_like(a, dtype=dtype)
