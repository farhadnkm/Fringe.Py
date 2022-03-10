import numpy as np
import os
from ..utils.io import export_image


def simulate_single(amplitude, phase, z, solver, export_path=None):
    """
    Simulates holograms by propagating a complex_valued tensor which is initialized with two images as phase and amplitude.

    Parameters
    ----------
    amplitude : array_like, ndarray, tensor - dtype: float32
        Amplitude image.

    phase : array_like, ndarray, tensor - dtype: float32
        Phase image.

    z : float
        Hologram plane.

    solver : object, class instance
        Optical propagation solver instance.

    export_path : string
        Path of the output image with a name and extension. If None, doesn't export. Default is None.

    Example
    ----------
    >>> p = ImageToArray(bit_depth=8, channel='gray', crop_window=None, dtype='float32')
    >>> amp = import_image("PATH_TO_IMAGE.png", preprocessor=p)
    >>> ph = import_image("PATH_TO_IMAGE.png", preprocessor=p)
    >>> solvers = AsSolver(shape=obj.shape, dr=1.12, backend="numpy")
    >>> simulate_single(amp, ph, 300, solvers, 'OUT_PATH/Img.tif')

    :return: Saves and returns the propagated holograms in the specified address.
    """

    obj = amplitude * np.exp(1j * phase * 2 * np.pi)
    h = np.square(np.abs(solver.solve(obj, z)))
    # rec_phase = unwrap_phase(np.angle(solvers.reconstruct(obj, z)))
    if export_path is not None:
        export_image(h, os.path.join(export_path), dtype='uint16')
    return h


def simulate_multiple(amplitude, phase, z, dz, count, solver, export_path):
    """
    Simulates a sequence of holograms by propagating a complex_valued tensor to a series of heights.

    Parameters
    ----------
    amplitude : array_like, ndarray, tensor - dtype: float32
        Amplitude image.

    phase : array_like, ndarray, tensor - dtype: float32
        Phase image.

    z : float
        Hologram plane.

    dz : float
        delta-height for each step.

    count: int
        The number of holograms with unique heights.

    solver : object, class instance
        Optical propagation solver instance.

    export_path : string
        Path of the output image with a name and extension. If None, doesn't export. Default is None.

    Example
    ----------
    >>> p = ImageToArray(bit_depth=8, channel='gray', crop_window=None, dtype='float32')
    >>> amp = import_image("PATH_TO_IMAGE.png", preprocessor=p)
    >>> ph = import_image("PATH_TO_IMAGE.png", preprocessor=p)
    >>> solvers = AsSolver(shape=obj.shape, dr=1.12, backend="numpy")
    >>> simulate_multiple(amp, ph, 300, 5, solvers, '~/OUT_DIR')

    :return: Saves and returns the propagated holograms in the specified directory.
    """

    obj = amplitude * np.exp(1j * phase * 2 * np.pi)
    hs = []
    for i in range(count):
        z_ = z + i * dz
        h = np.square(np.abs(solver.solve(obj, z_)))
        hs.append(h)
        if export_path is not None:
            export_image(h, os.path.join(export_path, str(i) + '_' + str(z_) + '.tif'), dtype='uint16')
    return hs
