import numpy as np
from skimage.restoration import unwrap_phase
from Utils.Processing_Np import AngularSpectrumSolver as AsSolver
from Utils.DataManager import import_image, export_image
from Utils.Modifiers import ImageToArray, PreprocessHologram
import os

import matplotlib
matplotlib.use("TkAgg")

pp1 = ImageToArray(bit_depth=16, channel='gray', crop_window=None, dtype='float32')
bg = import_image("PATH_TO_IMAGE.png", preprocessor=pp1)
pp2 = PreprocessHologram(background=bg)
h = import_image("PATH_TO_IMAGE.png", preprocessor=[pp1, pp2])

obj = h + 0j
solver = AsSolver(shape=obj.shape, dx=1.12, dy=1.12, wavelength=532e-3)

zs = list(range(-100, -400, -5))
export_directory = "PATH"

for i, z in enumerate(zs):
	res_amp = np.abs(solver.reconstruct(obj, z))
	res_phase = unwrap_phase(np.angle(solver.reconstruct(obj, z)))
	res_amp /= np.max(res_amp)
	res_amp *= 255
	export_image(res_amp, os.path.join(export_directory, str(i) + '_' + str(z) + '.png'))
