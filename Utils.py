import numpy as np

class PreprocessHologram:
	def __init__(self, bit_depth=16, background=None):
		self.bd = bit_depth
		self.bg = background

	def process(self, img):
		img = np.sqrt(img)
		img /= 2 ** self.bd
		if self.bg is not None:  # Normalize
			img /= self.bg
			minh = np.min(img)
			img -= minh
			img /= 1 - minh
		return img
