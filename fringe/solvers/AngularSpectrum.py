from typing import Union
from . import base
from ..backend.core import CoreFunctions
from ..backend.Numpy import Numpy
from ..backend.TensorFlow import TensorFlow
import numpy
_PI = numpy.pi


class AngularSpectrumSolver(base.Solver):
    def __init__(self, shape, dr: Union[float, tuple, list], is_batched, padding: Union[str, list, None] = None, pad_fill_value=0, backend='TensorFlow'):
        """
        Angular Spectrum Solver class. On initialization, static parameters are defined. To propagate fields, call the
        "solve" method.

        Parameters
        ----------
        shape : array_like, list, tuple, ndarray
            Array exhibiting input/output shape.

        dr : float, list[float], tuple[float]
            Pixel size in every dimension.

        is_batched : bool
            Set true if the first dimension of the input field is the batch dimension.

        padding : string, array_like, list[list[float]], ndarray
            An array storing the number of pixels to pad from the edges on each axis. It can be
            simply set as "SAME" to double the area of the input tensor by constant values. Default is None.
            Supported array format for a 2D field is [(before_0, after_0), (before_1, after_1)]

        pad_fill_value : float
            Constant fill value for padding. Default is 0.

        backend : string, class(backend.core.CoreFunctions)
            Computation backend. "tensorflow" and "numpy" are built in backends. Any custom class inherited
             from backend.core.CoreFunctions could be compatible as well.
        """
        if isinstance(backend, str):
            self.backend = {'tensorflow': TensorFlow,
                            'numpy': Numpy}[backend.lower()]
        elif issubclass(backend, CoreFunctions):
            self.backend = backend
        else:
            raise ValueError("The given backend is not an instance of backend.core.CoreFunctions or is not one of the "
                             "built-in classes.")

        if len(shape) > 2 and not is_batched:
            raise ValueError("More than 2-dimensional data structure is not supported without a batch dimension.")
        elif len(shape) > 3:
            raise ValueError("More than 3-dimensional data structure is not supported.")
        if len(shape) == 1 and is_batched:
            raise ValueError("The given shape is one dimensional while a batch dimension is also expected.")

        if type(padding) in (list, tuple):
            if len(padding) != len(shape) - (1 if is_batched else 0):
                raise ValueError("Padding must have the same length as the data dimensions. For each data dimension "
                                 "(except batch) a pad array must be defined.")

        self.pad_fill_value = pad_fill_value
        self.padding = []
        self.dr = []
        self.shape_f = []

        self.data_dim = []
        self.data_length = 0

        for i, s in enumerate(shape):
            if i != 0 or not is_batched:
                if type(padding) == str:
                    if padding.lower() == "same":
                        self.padding += [[s // 2] * 2]
                elif type(padding) in [list, tuple]:
                    self.padding += padding[i]
                else:
                    self.padding += [[0, 0]]

                self.dr += [dr] if (type(dr) is float or type(dr) is int) else [dr[i]]
                self.shape_f.append(shape[i] + self.padding[i][0] + self.padding[i][1])
                self.data_dim.append(i)
                self.data_length += 1

            elif i == 0 and is_batched:
                self.padding += [[0, 0]]

        kt = []
        for i in range(len(self.data_dim)):
            d = float(self.dr[i])
            n = self.shape_f[i]
            u = 1 / (d * n)
            if n % 2 == 0:
                f_min = -(n/2) * u
                f_max = (n/2 - 1) * u
            else:
                f_min = -((n - 1)/2) * u
                f_max = ((n - 1)/2) * u
            k_min, k_max = f_min * 2 * _PI, f_max * 2 * _PI
            kt.append(self.backend.fftshift(self.backend.linspace(k_min, k_max, n)))

        kt_grid = self.backend.meshgrid(*kt, indexing='ij') if len(self.data_dim) > 1 else kt
        self.kt_abs = self.backend.convert(kt_grid, dtype='float32')

        self.kt2 = self.backend.reduce_sum(self.backend.multiply(self.kt_abs, self.kt_abs), axis=0)
        self.dr = self.backend.convert(self.dr, dtype='float32')
        self.shape_f = self.backend.convert(self.shape_f, dtype='float32')

        for i, _ in enumerate(shape):
            if i == 0 and is_batched:
                self.kt2 = self.backend.expand_dim(self.kt2, i)
                self.kt_abs = self.backend.expand_dim(self.kt_abs, i + 1)

            self.dr = self.backend.expand_dim(self.dr, i + 1)
            self.shape_f = self.backend.expand_dim(self.shape_f, i + 1)

        self.t = 0

    def band_limit_mask(self, k, z):
        """
        Band limit in the Fourier domain For Angular Spectrum.

        Note:
            according to the paper ["Band-limited angular spectrum"], kx_limit requires a multiplier of 2.
            But without that (tighter band), results are more accurate.

        Parameters
        ----------
        k:  float
            Wave number : 2πn/λ

        z:  float
            Axial coordinate of the target plane.

        :return: A boolean mask which represents a low_pass filter in the frequency domain.
        """

        #k_limit = k / (2 * _PI * self.backend.sqrt((2 * z / (self.dr * self.shape_f)) ** 2 + 1))
        k_limit = k / (self.backend.sqrt((2 * z / (self.dr * self.shape_f)) ** 2 + 1))

        mask_2 = self.backend.where(self.backend.abs(self.kt_abs) < k_limit, 1, 0)
        mask = mask_2[0]
        for i in range(1, self.kt_abs.shape[0]):
            mask *= mask_2[i]

        return mask

    def propagator(self, k, z):
        """
        Free-space propagation phase shift : exp(iz x k_z) = exp(iz x √(k^2 - k_x^2 - k_y^2))

        Parameters
        ----------
        k: float
            Wave number : 2πn/λ

        z: float
            Axial coordinate of the target plane.

        :return: Complex-valued free-space propagator tensor with the shape of the input tensor (independent of batch size).
        """
        k2_kt2 = k * k - self.kt2
        sqk2_kt2 = self.backend.sqrt(self.backend.abs(k2_kt2))
        cs = self.backend.where(k2_kt2 >= 0,
                                self.backend.complex(real=self.backend.zeros_like(self.kt2), imag=sqk2_kt2 * z),
                                self.backend.complex(real=-sqk2_kt2 * z, imag=self.backend.zeros_like(self.kt2)))
        return self.backend.exp(cs)

    def transfer_function(self, k, z):
        """
        Complex Optical Transfer Function (OTF) which here, is a low-pass filtered version of the propagator function.

        Parameters
        ----------
        k: float
            Wave number : 2πn/λ

        z: float
            Axial coordinate of the target plane.

        :return: Complex-valued OTF tensor with the shape of the input tensor (independent of batch size).
        """
        mask = self.band_limit_mask(k, z)
        p = self.propagator(k, z)
        p_m = self.backend.where(mask > 0, p, 0)
        return p_m

    def solve(self, input_, k, z):
        """
        Solves convolution of the complex input field with the angular spectrum optical transfer function for a given
        wave number k, and axial displacement z.

        Parameters
        ----------
        input_: ndarray, tensor - dtype: complex64
            Complex input field.

        k: float
            Wave number : 2πn/λ

        z: float
            Axial coordinate of the target plane.

        :return: Complex-valued angular spectrum of the input field.
        """
        if len(input_.shape) != len(self.padding):
            raise ValueError("Input shape is incompatible.")

        field = self.backend.pad(input_, self.padding, self.pad_fill_value)

        tf = self.transfer_function(k, z)
        return self.backend.unpad(self.backend.ifft(tf * self.backend.fft(field, self.data_length), self.data_length), self.padding)
