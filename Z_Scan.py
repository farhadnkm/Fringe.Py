import numpy as np
from skimage.restoration import unwrap_phase
from Utils.Processing import AngularSpectrumSolver as AsSolver
from Utils.DataManager import import_image, export_image
from Utils.Modifiers import ImageToArray, PreprocessHologram
import os

import matplotlib
matplotlib.use("TkAgg")

pp1 = ImageToArray(bit_depth=16, channel='gray', crop_window=[450, 400, 256, 256], dtype='float32')
bg = import_image("C:/Users/e-min/Downloads/Compressed/5.10.99/BG2.png", preprocessor=pp1)
pp2 = PreprocessHologram(background=bg)
h = import_image("C:/Users/e-min/Downloads/Compressed/hemoglobinh1/18 27 1200.png", preprocessor=[pp1, pp2])

obj = h + 0j
solver = AsSolver(shape=obj.shape, dx=1.12, dy=1.12, wavelength=650e-3) #, wavelength=532e-3)

zs = list(range(-110, -150, -1))
export_directory = "C:/Users/e-min/Downloads/Compressed/exports"

for i, z in enumerate(zs):
	res_amp = np.abs(solver.reconstruct(obj, z))
	res_phase = unwrap_phase(np.angle(solver.reconstruct(obj, z)))
	res_amp /= np.max(res_amp)
	res_amp *= 255
	export_image(res_amp, os.path.join(export_directory, str(i) + '_' + str(z) + '.png'))

#fig, (ax1, ax2) = plt.subplots(1, 2)
#ax1.imshow(res_amp, cmap='gray')
#ax2.imshow(res_phase, cmap='gray')
#plt.show()

