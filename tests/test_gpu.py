import numpy as np
import os
from skimage.restoration import unwrap_phase
from data.io import import_image, import_image_seq, export_image
from data.modifiers import ImageToArray, PreprocessHologram, ConvertToTensor
from process.gpu import AngularSpectrumSolver as AsSolver, MultiHeightPhaseRecovery as MHPR
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt

device = "gpu"

if device == "gpu":
	if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
		print('GPU is up and running')
		device = "/gpu:0"
	else:
		print('GPU is not available. The program will run on CPU')
		device = "/cpu:0"
elif device == "tpu":
	if len(tf.config.experimental.list_physical_devices('TPU')) > 0:
		print('TPU is up and running')
		device = "/tpu:0"
	else:
		print('TPU is not available. The program will run on CPU')
		device = "/cpu:0"
else:
	device = "/cpu:0"

dtype_f = tf.float32
dtype_c = tf.complex64

h_path = "docs/images/sequence/0_300.tif"
p1 = ImageToArray(bit_depth=16, channel='gray', crop_window=None, dtype='float32')
p2 = ConvertToTensor(dtype=dtype_c)
bg = np.ones((512, 512))
p3 = PreprocessHologram(background=bg)
h = import_image(h_path, preprocessor=[p1, p2, p3])

z = -300

solver = AsSolver(shape=h.shape, dx=1.12, dy=1.12, wavelength=532e-3)

amp = np.abs(solver.reconstruct(h, z))
phase = unwrap_phase(np.angle(solver.reconstruct(h, z)))

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
p2 = ConvertToTensor(dtype=dtype_c)
bg = np.ones((512, 512))
p3 = PreprocessHologram(background=bg)
h_seq = import_image_seq(img_seq_paths, preprocessor=[p1, p2, p3])

z_values = [-300, -350, -400, -450, -500, -550, -600, -650]
iterations = 100

solver = AsSolver(shape=h_seq[0].shape, dx=1.12, dy=1.12, wavelength=532e-3, dtype_f=dtype_f, dtype_c=dtype_c)

with tf.device(device):
	mhpr = MHPR(solver)
	recovered_h = mhpr.resolve(h_seq, z_values, iterations)
	amp = np.abs(recovered_h)
	phase = unwrap_phase(np.angle(recovered_h))
	phase += np.pi
	phase /= 2 * np.pi

	plt.imshow(amp, cmap='gray')
	plt.show()
	plt.imshow(phase, cmap='viridis')
	plt.show()

cmap = matplotlib.cm.get_cmap('viridis')

export_image(amp, os.path.join('docs/images/exports', 'amplitude.png'), dtype='uint8')
export_image(phase, os.path.join('docs/images/exports', 'phase.png'), dtype='uint8')
export_image(cmap(phase), os.path.join('docs/images/exports', 'phase_colored.png'), dtype='uint8')

