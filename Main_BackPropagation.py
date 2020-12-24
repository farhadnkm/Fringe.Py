import numpy as np
from skimage.restoration import unwrap_phase
from Processing import AngularSpectrumSolver as AsSolver
from Utils import PreprocessHologram
from FileManager import import_image

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


bg = import_image("PATH_TO_IMAGE.tif", dims=[850, 1000, 512, 512])
pph = PreprocessHologram(bit_depth=16, background=bg)
h = import_image("PATH_TO_IMAGE.tif", dims=[850, 1000, 512, 512], preprocessor=pph)

obj = h + 0j
z = -306
solver = AsSolver(shape=obj.shape, dx=1.12, dy=1.12, wavelength=532e-3)

res_amp = np.abs(solver.reconstruct(obj, z))
res_phase = unwrap_phase(np.angle(solver.reconstruct(obj, z)))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(res_amp, cmap='gray')
ax2.imshow(res_phase, cmap='gray')
plt.show()

