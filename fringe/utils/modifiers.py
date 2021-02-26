import numpy as np
from skimage import color
import tensorflow as tf


# Modifiers are classes having a 'process()' function. These functions could be passed as a parameter
# to import_image methods in DataManager and are called on each image on import


class ImageToArray:
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

    def process(self, img):
        if len(np.shape(img)) == 2:
            if self.chan in ['r', 'g', 'b', 'rgb']:
                raise AssertionError('The image is grayscale but your expecting an RGB image.')

        _img = img.astype(self.dtype)
        _img /= 2 ** self.bd

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

        return _img


class PreprocessHologram:
    def __init__(self, background=None):
        """
        A preprocess class to convert the input image to a hologram for further processing and reconstruction.
        :param background: The background image of the hologram.
        """
        self.bg = background
        self.bg[self.bg <= 1e-8] = 1e-8

    def process(self, img):
        _img = np.copy(img)
        if self.bg is not None:  # Normalize
            _img /= self.bg
        minh = np.min(_img)
        _img -= minh
        _img /= 1 - minh
        return _img


class ConvertToTensor:
    def __init__(self, dtype='float32'):
        """
        :param dtype: Data type of the resulting tensor
        """
        assert dtype in ['float32', 'float64', 'complex64', 'complex128']
        self.dtype = dtype

    def process(self, img):
        if self.dtype == 'complex64':
            t = tf.complex(real=tf.convert_to_tensor(img, dtype='float32'),
                           imag=tf.zeros_like(img, dtype='float32'))
        elif self.dtype == 'complex128':
            t = tf.complex(real=tf.convert_to_tensor(img, dtype='float64'),
                           imag=tf.zeros_like(img, dtype='float64'))
        elif self.dtype in ['float32', 'float64']:
            t = tf.convert_to_tensor(img, dtype=self.dtype)
        else:
            raise TypeError(str(self.dtype) + ' is not supported')
        return t
