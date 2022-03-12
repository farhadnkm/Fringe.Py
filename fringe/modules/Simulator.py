import numpy as np
import os
from ..utils.io import export_image


def simulate_multiple(input_field, k, z, dz, count, solver, export_path):
    """
    Simulates a sequence of holograms by propagating a complex_valued tensor to a series of heights.

    Parameters
    ----------
    input_field :  array_like - dtype: complex64
        The input complex field.

    k: float
        Wave number

    z : float
        Hologram plane.

    dz : float
        delta-height for each step.

    count: int
        The number of holograms with unique heights.

    solver : class(solvers.base.Solver)
        The propagation solver class which encapsulates Solver.solve(input_, *args) to apply propagation
        algorithm on the input field.

    export_path : string
        Path of the output image with a name and extension. If None, doesn't export. Default is None.

    Example
    ----------
    >>> field = np.zeros((128, 128)).astype('complex64')
    >>> field[50:55, 90:100] = 1 + 0j
    >>> solver = AngularSpectrumSolver(shape=obj.shape, dr=2, padding=None, backend="numpy")
    >>> simulate_multiple(field, 2*PI/500e-3, 300, 5, solver, '~/OUT_DIR')

    :return: Saves and returns the propagated holograms in the specified directory.
    """

    hs = []
    for i in range(count):
        z_ = z + i * dz
        h = np.square(np.abs(solver.solve(input_field, k, z_)))
        hs.append(h)
        if export_path is not None:
            export_image(h, os.path.join(export_path, str(i) + '_' + str(z_) + '.tif'), dtype='uint16')
    return hs
