import numpy as np
from skimage.restoration import unwrap_phase
from Processing import AngularSpectrumSolver as AsSolver, MultiHeightPhaseRecovery as MHPR
from FileManager import import_image, import_image_seq
from Utils import PreprocessHologram

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

bg_path = "PATH_TO_IMAGE.tif"

img_seq_paths = [
	"PATH_TO_IMAGE/1.tif",
	"PATH_TO_IMAGE/2.tif",
	"PATH_TO_IMAGE/3.tif",
	"PATH_TO_IMAGE/4.tif",
	"PATH_TO_IMAGE/5.tif",
	"PATH_TO_IMAGE/6.tif",
	"PATH_TO_IMAGE/7.tif",
	"PATH_TO_IMAGE/8.tif"
]

bg = import_image(bg_path, dims=[1100, 1200, 512, 512],
				  preprocessor=PreprocessHologram(bit_depth=16))

h_seq = import_image_seq(img_seq_paths, dims=[1100, 1200, 512, 512],
						 preprocessor=PreprocessHologram(bit_depth=16, background=bg))

solver = AsSolver(shape=h_seq[0].shape, dx=1.12, dy=1.12, wavelength=532e-3)
z_values = [306, 362, 431, 481, 543, 602, 674, 718]

mhpr = MHPR(solver)
recovered_h = mhpr.resolve(h_seq, z_values, 30)
amp = np.abs(recovered_h)
phase = unwrap_phase(np.angle(recovered_h))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(amp, cmap='gray')
ax2.imshow(phase, cmap='gray')
plt.show()