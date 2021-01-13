import numpy as np
from skimage.restoration import unwrap_phase
from Utils.Processing import AngularSpectrumSolver as AsSolver
from Utils.DataManager import import_image
from Utils.Modifiers import ImageToArray, PreprocessHologram

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

pp1 = ImageToArray(bit_depth=16, channel='gray', crop_window=None, dtype='float32')
bg = import_image("D:/GoogleDrive_/Colab/Dataset/Custom/tests/baboon.png", preprocessor=pp1)
pp2 = PreprocessHologram(background=bg)
h = import_image("D:/GoogleDrive_/Colab/Dataset/Custom/tests/peppers.png", preprocessor=[pp1, pp2])

obj = h + 0j
solver = AsSolver(shape=obj.shape, dx=1.12, dy=1.12, wavelength=532e-3)

z = -306
res_amp = np.abs(solver.reconstruct(obj, z))
res_phase = unwrap_phase(np.angle(solver.reconstruct(obj, z)))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(res_amp, cmap='gray')
ax2.imshow(res_phase, cmap='gray')
plt.show()

