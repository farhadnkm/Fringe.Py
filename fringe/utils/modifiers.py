import numpy as np
from skimage import color
import tensorflow as tf
from abc import abstractmethod

# Modifiers are classes having a 'solvers()' function. These functions could be passed as a parameter
# to import_image methods in DataManager and are called on each image on import


class Modifier:
    @abstractmethod
    def process(self, input_):
        raise NotImplementedError


class ImageToArray(Modifier):
    def __init__(self, bit_depth=16, channel='gray', crop_window=None, dtype='float32'):
        """
        A preprocess class to convert the input image to a valid array for further processing.
        :param bit_depth: bit_depth of the input image.
        :param channel: Preferred channel of image for the processing: could be 'gray', 'r', 'g', 'b', 'avg_rgb'.
        :param crop_window: [x, y, width, height] specifies the crop window. If is none, the actual size is considered.
        :param dtype: Data type of the output array.
        """
        self.bd = bit_depth
        assert channel in ['gray', 'r', 'g', 'b', 'rgb']
        self.chan = channel
        self.crop = crop_window
        self.dtype = dtype

    def process(self, img, *args, **kwargs):
        if len(np.shape(img)) == 2:
            if self.chan in ['r', 'g', 'b', 'rgb']:
                raise AssertionError('The image is grayscale but your expecting an RGB image.')

        _img = img.astype('float32')
        _img = _img / (2 ** self.bd - 1)

        if self.chan == 'gray':
            if len(np.shape(_img)) == 2:
                pass
            elif len(np.shape(_img)) == 3:
                _img = color.rgb2gray(_img)

            else:
                raise ValueError('Input array is not an image or its type is not supported.')
        elif self.chan == 'r':
            _img = _img[:, :, 0]
        elif self.chan == 'g':
            _img = _img[:, :, 1]
        elif self.chan == 'b':
            _img = _img[:, :, 2]
        elif self.chan == 'rgb':
            pass

        if self.crop is not None:
            x, y, w, h = self.crop
            _img = _img[y:y + h, x:x + w]

        _img = _img.astype(self.dtype)
        return _img


class Normalize(Modifier):
    def __init__(self, background=None):
        """
        A preprocess class to convert the normalize images based on their background.
        :param background: The background image of the hologram.
        """
        self.bg = background
        self.bg[self.bg <= 1e-8] = 1e-8

    def process(self, img, *args, **kwargs):
        _img = np.copy(img)
        if self.bg is not None:
            _img /= self.bg
        minh = np.min(_img)
        _img -= minh
        _img /= 1 - minh
        return _img


class MakeComplex(Modifier):
    def __init__(self, dtype='complex64', set_as='amplitude', **kwargs):
        """
        Generates a complex array from the imported image

        Parameters
        ----------
        dtype : string
            Output data type : 'complex64' - 'complex128'. Default is 'complex64'.

        set_as : string
            Determines whether the imported image take part as 'real', 'imaginary',
            'amplitude', or 'phase'. Default is 'amplitude'.

        real : float ndarray (Optional)
            fills real part when image is set as 'imaginary'. Default is 1.

        imaginary : float, ndarray (Optional)
            fills imaginary part when image is set as 'real'. Default is 0.

        amplitude : float, ndarray (Optional)
            Overrides amplitude term when image is set as 'phase'. Default is 1.

        phase : float, ndarray (Optional)
            Overrides phase term when image is set as 'amplitude'. Default is 0.

        phase_coef : float (Optional)
            phase coefficient. For example, could be set to 2pi to scale up
            normalized phase to cover full radians. Default is 1.
        """

        assert dtype in ['complex64', 'complex128']
        assert set_as in ['real', 'imaginary', 'amplitude', 'phase']
        self.dtype = dtype
        self.target = set_as
        self._imag = kwargs.get('imaginary', 0)
        self._real = kwargs.get('real', 1)
        self._ph = kwargs.get('phase', 0)
        self._amp = kwargs.get('amplitude', 1)
        self._ph_coef = kwargs.get('phase_coef', 1)

    def process(self, img, *args, **kwargs):
        if self.target == 'real':
            return (img + 1j * self._imag).astype(self.dtype)

        elif self.target == 'imaginary':
            return (self._real + 1j * img).astype(self.dtype)

        elif self.target == 'amplitude':
            return (img * np.exp(1j * self._ph)).astype(self.dtype)

        elif self.target == 'phase':
            return (self._amp * np.exp(1j * img * self._ph_coef)).astype(self.dtype)


class ConvertToTensor(Modifier):
    def __init__(self):
        """
        Converts imported arrays to tensorflow tensors.
        """
        pass

    def process(self, img, *args, **kwargs):
        return tf.convert_to_tensor(img)


class Map(Modifier):
    def __init__(self, function):
        """
        Converts imported arrays to tensorflow tensors.
        """
        self.function = function
        pass

    def process(self, img, *args, **kwargs):
        return self.function(img, *args, **kwargs)
