import numpy as np
import os
from skimage.restoration import unwrap_phase
from Utils.Processing import AngularSpectrumSolver as AsSolver
from Utils.DataManager import import_image, export_image
from Utils.Modifiers import ImageToArray, PreprocessHologram

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

pp1 = ImageToArray(bit_depth=8, channel='gray', crop_window=None, dtype='float32')
amp = import_image("D:/Research data/results/images/selected/Simulation/amplitude.png", preprocessor=pp1)
ph = import_image("D:/Research data/results/images/selected/Simulation/phase.png", preprocessor=pp1)

obj = amp * np.exp(1j * ph * 2 * np.pi)
solver = AsSolver(shape=obj.shape, dx=1.12, dy=1.12, wavelength=532e-3)

dz = 50
exp_root_path = 'D:/Research data/results/images/selected/Simulation/test 1/MHPR/Holograms'
for i in range(8):
	z = 300 + i * dz
	rec_int = np.square(np.abs(solver.reconstruct(obj, z)))
	export_image(rec_int, os.path.join(exp_root_path, str(i) + '_' + str(z) + '.tif'), dtype='uint16')
