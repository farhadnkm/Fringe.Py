import numpy as np
from skimage.restoration import unwrap_phase
from Processing import AngularSpectrumSolver as AsSolver
from Utils import PreprocessHologram
from FileManager import import_image, export_image

import os

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


bg = import_image("C:/Users/e-min/Downloads/Compressed/5.10.99/BG2.png", dims=[450, 400, 256, 256],
				  preprocessor=PreprocessHologram(bit_depth=16))
h = import_image("C:/Users/e-min/Downloads/Compressed/hemoglobinh1/18 27 1200.png", dims=[450, 400, 256, 256],
				 preprocessor=PreprocessHologram(bit_depth=16, background=bg))

obj = h + 0j
solver = AsSolver(shape=obj.shape, dx=1.12, dy=1.12, wavelength=650e-3) #, wavelength=532e-3)

zs = list(range(-120, -130, -1))
export_directory = "C:/Users/e-min/Downloads/Compressed/exports"

for i in range(len(zs)):
	res_amp = np.abs(solver.reconstruct(obj, zs[i]))
	res_phase = unwrap_phase(np.angle(solver.reconstruct(obj, zs[i])))
	res_amp /= np.max(res_amp)
	res_amp *= 255
	export_image(res_amp, os.path.join(export_directory, str(zs[i]) + '.png'))

#fig, (ax1, ax2) = plt.subplots(1, 2)
#ax1.imshow(res_amp, cmap='gray')
#ax2.imshow(res_phase, cmap='gray')
#plt.show()

