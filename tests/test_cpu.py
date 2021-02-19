import numpy as np
import os
from skimage.restoration import unwrap_phase
from data.io import import_image, import_image_seq, export_image
from data.modifiers import ImageToArray, PreprocessHologram
from process.cpu import AngularSpectrumSolver as AsSolver, MultiHeightPhaseRecovery as MHPR
import matplotlib

h_path = "docs/images/sequence/0_300.tif"
p1 = ImageToArray(bit_depth=16, channel='gray', crop_window=None, dtype='float32')
bg = np.ones((512, 512))  # if no background
p2 = PreprocessHologram(background=bg)
h = import_image(h_path, preprocessor=[p1, p2])

z = -300

obj = h + 0j
solver = AsSolver(shape=obj.shape, dx=1.12, dy=1.12, wavelength=532e-3)

rec = solver.solve(obj, z)
amp = np.abs(rec)
phase = unwrap_phase(np.angle(rec))


img_seq_paths = [
	"docs/images/sequence/0_300.tif",
	"docs/images/sequence/1_350.tif",
	"docs/images/sequence/2_400.tif",
	"docs/images/sequence/3_450.tif",
	"docs/images/sequence/4_500.tif",
	"docs/images/sequence/5_550.tif",
	"docs/images/sequence/6_600.tif",
	"docs/images/sequence/7_650.tif"
]

p1 = ImageToArray(bit_depth=16, channel='gray', crop_window=None, dtype='float32')
bg = np.ones((512, 512))
p2 = PreprocessHologram(background=bg)
h_seq = import_image_seq(img_seq_paths, preprocessor=[p1, p2])

z_values = [-300, -350, -400, -450, -500, -550, -600, -650]
iterations = 100

solver = AsSolver(shape=h_seq[0].shape, dx=1.12, dy=1.12, wavelength=532e-3)

mhpr = MHPR(solver)
recovered_h = mhpr.solve(h_seq, z_values, iterations)
amp = np.abs(recovered_h)
phase = unwrap_phase(np.angle(recovered_h))
phase += np.pi
phase /= 2 * np.pi

cmap = matplotlib.cm.get_cmap('viridis')
