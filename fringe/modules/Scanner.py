import numpy as np
from ..utils.io import export_image
import os


def scan_z(input_field, k, z_range, dz, solver, export_dir):
    """
    Scans the z axis within the specified range with the specified dz resolution and saves the propagated images in the
    given directory. This method is suitable to find the focus plane of a hologram.

    Parameters
    ----------
    input_field :  array_like - dtype: complex64
        The input complex field.

    k : floar
        Wave number.

    z_range :   array_like
        The range of z values: (min_z, max_z)

    dz : float
        Step size of scanning.

    solver : class(solvers.base.Solver)
        The propagation solver class which encapsulates Solver.solve(input_, *args) to apply propagation
        algorithm on the input field.

    export_dir : string
        Output directory to export images.

    Example
    ----------
    >>> field = np.zeros((128, 128)).astype('complex64')
    >>> field[50:55, 90:100] = 1 + 0j
    >>> solver = AngularSpectrumSolver(shape=obj.shape, dr=1, padding=None, backend="numpy")
    >>> scan_z(field, 2*PI/500e-3, (-300, -500), -5, solver, '~/OUT_DIR')

    :return: Saves the reconstructed images in the given directory with the order of their heights.
    """
    np.abs()
    obj = input_field + 0j
    zs = list(range(z_range[0], z_range[1], dz))
    for i, z in enumerate(zs):
        res_amp = np.abs(solver.solve(obj, k, z))
        # res_phase = unwrap_phase(np.angle(solvers.reconstruct(obj, z)))
        res_amp /= np.max(res_amp)
        res_amp *= 255
        export_image(res_amp, os.path.join(export_dir, str(i) + '_' + str(z) + '.png'))
