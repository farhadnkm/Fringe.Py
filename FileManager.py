import gdal
from PIL import Image
import numpy as np


def import_image(path, dims=None, preprocessor=None):
	"""
	:param path: Directory of the image
	:param dims: [x, y, width, height] describes the crop window
	:return: a numpy array containing the image
	"""
	img = gdal.Open(path).ReadAsArray().astype('float32')
	print("Imported image path:", path)
	if dims is not None:
		img = img[dims[1]:dims[1] + dims[3], dims[0]:dims[0] + dims[2]]
	if preprocessor is not None:
		img = preprocessor.process(img=img)
	return img


def import_image_seq(paths, dims=None, preprocessor=None):
	"""
	:param paths: Directories of the images
	:param dims: [x, y, width, height] specifies the crop window. If none returns the actual size
	:param preprocessor: An arbitrary class having a function 'process' which is performed on the images on import
	:return: A list numpy arrays
	"""
	imgs = []
	for path in paths:
		img = import_image(path, dims, preprocessor)
		imgs.append(img)
	return imgs


def export(img, path):
	out_amp = np.uint8(img)
	out_amp = Image.fromarray(out_amp)
	out_amp = out_amp.convert('L')
	out_amp.save(path)
