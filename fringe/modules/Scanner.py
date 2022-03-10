import numpy as np
from ..utils.io import export_image
import os


def scan_z(hologram, z_range, dz, solver, export_dir):
    """
    Scans the z axis within the specified range with the specified dz resolution and saves the propagated images in the
    given directory. This method is suitable to find the focus plane of a hologram.

    Parameters
    ----------
    hologram :  array_like - dtype: complex64
        Input hologram image.
    z_range :   array_like
        The range of z values: (min_z, max_z)
    dz : float
        Step size of scanning.
    export_dir : string
        Output directory to export images.

    Example
    ----------
    >>> p1 = ImageToArray(bit_depth=16, channel='gray', crop_window=None, dtype='float32')
    >>> bg = import_image("PATH.tif", preprocessor=p1)
    >>> p2 = PreprocessHologram(background=bg)
    >>> h = import_image("PATH.tif", preprocessor=[p1, p2])
    >>> solvers = AsSolver(shape=obj.shape, dr=1.12, backend="numpy")
    >>> scan_z(h, (-300, -500), -5, solvers, '~/OUT_DIR')


    :return: Saves the reconstructed images in the given directory with the order of their heights.
    """
    np.abs()
    obj = hologram + 0j
    zs = list(range(z_range[0], z_range[1], dz))
    for i, z in enumerate(zs):
        res_amp = np.abs(solver.reconstruct(obj, z))
        # res_phase = unwrap_phase(np.angle(solvers.reconstruct(obj, z)))
        res_amp /= np.max(res_amp)
        res_amp *= 255
        export_image(res_amp, os.path.join(export_dir, str(i) + '_' + str(z) + '.png'))
