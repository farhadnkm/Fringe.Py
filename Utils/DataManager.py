import numpy as np
from skimage import io
import tensorflow as tf

def import_image(path, preprocessor=None):
	"""
	:param path: Directory of the image
	:param preprocessor: An arbitrary class having a function 'process' which is performed on the images on import
	:return: a numpy array containing the image
	"""
	img = io.imread(path)

	print("Image imported from:", path)
	if preprocessor is not None:
		if hasattr(preprocessor, '__iter__'):
			for pp in preprocessor:
				img = pp.process(img=img)
		else:
			img = preprocessor.process(img=img)

	return img


def import_image_seq(paths, preprocessor=None):
	"""
	:param paths: Directories of the images
	:param preprocessor: An arbitrary class having a function 'process' which is performed on the images on import
	:return: A list of images as numpy arrays
	"""
	imgs = []
	for path in paths:
		img = import_image(path, preprocessor)
		imgs.append(img)
	return imgs

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
def export_image(img, path, dtype='uint16'):
	assert dtype in ['uint8', 'uint16']
	_img = img.copy()
	_img[_img > 1] = 1
	_img[_img < 0] = 0

	if dtype == 'uint8':
		_img *= 2 ** 8 - 1
		_img = np.uint8(_img)
	elif dtype == 'uint16':
		_img *= 2 ** 16 - 1
		_img = np.uint16(_img)

	io.imsave(path, _img)
	print("Image exported to:", path)


def convert_to_tensor(ndarray, dtype):
	if dtype == 'complex64':
		t = tf.complex(real=tf.convert_to_tensor(ndarray, dtype='float32'),
							imag=tf.zeros_like(ndarray, dtype='float32'))
	elif dtype == 'complex128':
		t = tf.complex(real=tf.convert_to_tensor(ndarray, dtype='float64'),
							imag=tf.zeros_like(ndarray, dtype='float64'))
	elif dtype in ['float32', 'float64']:
		t = tf.convert_to_tensor(ndarray, dtype=dtype)
	else:
		raise TypeError(str(dtype) + ' is not supported')
	return t


def convert_to_tensor_seq(seq, dtype):
	tensor_array = []
	for img in seq:
		t = convert_to_tensor(img, dtype)
		tensor_array.append(t)

	return tensor_array
