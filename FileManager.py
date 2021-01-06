import gdal
from PIL import Image
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
def import_image(path, dims=None, preprocessor=None):
	"""
	:param path: Directory of the image
	:param dims: [x, y, width, height] describes the crop window
	:return: a numpy array containing the image
	"""
	img = gdal.Open(path).ReadAsArray()

	#fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
	#ax1.imshow(img[0], cmap='gray')
	#ax2.imshow(img[1], cmap='gray')
	#ax3.imshow(img[2], cmap='gray')
	#plt.show()
	if len(np.shape(img)) == 3 and np.shape(img)[0] == 3:
		img = img[0].astype('float32')
	else:
		img = img.astype('float32')
	print("Image imported from:", path)
	if dims is not None:
		img = img[dims[1]:dims[1] + dims[3], dims[0]:dims[0] + dims[2]]
	if preprocessor is not None:
		img = preprocessor.process(img=img)
	return img


def import_image_seq(paths, dims=None, preprocessor=None):
	"""
	:param paths: Directories of the images
	:param dims: [x, y, width, height] specifies the crop window. If is none, the actual size is considered
	:param preprocessor: An arbitrary class having a function 'process' which is performed on the images on import
	:return: A list of images as numpy arrays
	"""
	imgs = []
	for path in paths:
		img = import_image(path, dims, preprocessor)
		imgs.append(img)
	return imgs


def export_image(img, path):
	print("Image exported to:", path)
	out_amp = np.uint8(img)
	out_amp = Image.fromarray(out_amp)
	out_amp = out_amp.convert('L')
	out_amp.save(path)
