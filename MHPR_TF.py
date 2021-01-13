import numpy as np
import os
from skimage.restoration import unwrap_phase
from Utils.Processing_GPU import AngularSpectrumSolver as AsSolver, MultiHeightPhaseRecovery as MHPR
from Utils.DataManager import import_image, import_image_seq, export_image, convert_to_tensor_seq
from Utils.Modifiers import ImageToArray, PreprocessHologram
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


bg_path = "PATH_TO_IMAGE.tif"

img_seq_paths = [
	"D:/Research data/results/images/selected/Simulation/test 1/MHPR/Holograms/0_300.tif",
	"D:/Research data/results/images/selected/Simulation/test 1/MHPR/Holograms/1_350.tif",
	"D:/Research data/results/images/selected/Simulation/test 1/MHPR/Holograms/2_400.tif",
	"D:/Research data/results/images/selected/Simulation/test 1/MHPR/Holograms/3_450.tif",
	"D:/Research data/results/images/selected/Simulation/test 1/MHPR/Holograms/4_500.tif",
	"D:/Research data/results/images/selected/Simulation/test 1/MHPR/Holograms/5_550.tif",
	"D:/Research data/results/images/selected/Simulation/test 1/MHPR/Holograms/6_600.tif",
	"D:/Research data/results/images/selected/Simulation/test 1/MHPR/Holograms/7_650.tif"
]


pp1 = ImageToArray(bit_depth=16, channel='gray', crop_window=None, dtype='float32')
#bg = import_image(bg_path, preprocessor=pp1)
bg = np.ones((512, 512))  # no background
pp2 = PreprocessHologram(background=bg)
h_seq = import_image_seq(img_seq_paths, preprocessor=[pp1, pp2])
h_seq = convert_to_tensor_seq(h_seq, dtype='complex64')


solver = AsSolver(shape=h_seq[0].shape, dx=1.12, dy=1.12, wavelength=532e-3, dtype_f='float32', dtype_c='complex64')
z_values = [300, 350, 400, 450, 500, 550, 600, 650]

mhpr = MHPR(solver)
recovered_h = mhpr.resolve(h_seq, z_values, 100)
amp = np.abs(recovered_h)
phase = unwrap_phase(np.angle(recovered_h))
phase += np.pi
phase /= 2 * np.pi


cmap = matplotlib.cm.get_cmap('viridis')

export_image(amp, os.path.join('D:/Research data/results/images/selected/Simulation/test 1/MHPR/', 'amp.png'), dtype='uint8')
export_image(phase, os.path.join('D:/Research data/results/images/selected/Simulation/test 1/MHPR/', 'ph.png'), dtype='uint8')
export_image(cmap(phase), os.path.join('D:/Research data/results/images/selected/Simulation/test 1/MHPR/', 'ph_cm.png'), dtype='uint8')


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(amp, cmap='gray')
ax2.imshow(phase, cmap='viridis')
plt.show()
