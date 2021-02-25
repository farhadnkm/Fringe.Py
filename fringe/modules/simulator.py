import numpy as np
import os
from ..utils.io import export_image


def simulate_single(amplitude, phase, z, solver, export_path):
    """
    Simulates holograms by propagating a complex_valued tensor which is initialized with two images as phase and amplitude.
    Example:
    >> p = ImageToArray(bit_depth=8, channel='gray', crop_window=None, dtype='float32')
    >> amp = import_image("PATH_TO_IMAGE.png", preprocessor=p)
    >> ph = import_image("PATH_TO_IMAGE.png", preprocessor=p)
    >> solver = AsSolver(shape=obj.shape, dx=1.12, dy=1.12, wavelength=532e-3)
    >> simulate_single(amp, ph, 300, solver, 'OUT_PATH/Img.tif')

    :param amplitude: The amplitude image.
    :param phase: The phase image.
    :param z: The hologram plane.
    :param solver: The holographic solver algorithm with proper settings.
    :param export_path: The path of the output image with the name and extension.
    :return: Saves the propagated holograms in the specified address.
    """

    obj = amplitude * np.exp(1j * phase * 2 * np.pi)
    hologram = np.square(np.abs(solver.reconstruct(obj, z)))
    # rec_phase = unwrap_phase(np.angle(solver.reconstruct(obj, z)))
    export_image(hologram, os.path.join(export_path), dtype='uint16')


def simulate_multiple(amplitude, phase, z, dz, count, solver, export_dir):
    """
    Simulates a sequence of holograms by propagating a complex_valued tensor to a series of heights.
    Example:
    >> p = ImageToArray(bit_depth=8, channel='gray', crop_window=None, dtype='float32')
    >> amp = import_image("PATH_TO_IMAGE.png", preprocessor=p)
    >> ph = import_image("PATH_TO_IMAGE.png", preprocessor=p)
    >> solver = AsSolver(shape=obj.shape, dx=1.12, dy=1.12, wavelength=532e-3)
    >> simulate_multiple(amp, ph, 300, 5, solver, '~/OUT_DIR')

    :param amplitude: The amplitude image.
    :param phase: The phase image.
    :param z: The reference hologram plane.
    :param dz: The delta_height between hologram planes.
    :param count: The number of holograms with unique heights.
    :param solver: The holographic solver algorithm with proper settings.
    :param export_dir: A directory address in which the holograms will be saved.
    :return: Saves the propagated holograms in the specified directory.
    """

    obj = amplitude * np.exp(1j * phase * 2 * np.pi)
    for i in range(count):
        z_ = z + i * dz
        hologram = np.square(np.abs(solver.reconstruct(obj, z_)))
        export_image(hologram, os.path.join(export_dir, str(i) + '_' + str(z_) + '.tif'), dtype='uint16')
