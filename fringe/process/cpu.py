import numpy as np
from numpy.fft import fft2, ifft2


class AngularSpectrumSolver:
    def __init__(self, shape, dx, dy, wavelength):
        self.shape = shape
        self.dx = dx
        self.dy = dy
        self.k = 2 * np.pi / wavelength
        self.k2 = self.k * self.k
        kx = np.fft.fftfreq(shape[0], dx / (2 * np.pi))
        ky = np.fft.fftfreq(shape[1], dy / (2 * np.pi))
        kx, ky = np.meshgrid(kx, ky, indexing='ij', sparse=True)
        self.kz2 = kx * kx + ky * ky
        self.valid_mask = self.k2 > self.kz2

    def propagator(self, z):
        p = np.zeros(self.shape, dtype=np.complex_)
        p[self.valid_mask] = np.exp(1j * np.sqrt(self.k2 - self.kz2[self.valid_mask]) * z)
        return p

    def solve(self, initializer, z):
        prop = self.propagator(z)
        return ifft2(prop * fft2(initializer))


class MultiHeightPhaseRecovery:
    def __init__(self, solver):
        self.solver = solver

    def solve(self, h_seq, z_values, iterations=20):
        recovered_h = h_seq[0] * np.exp(0j)

        for i in range(iterations):
            if i % 5 == 0:
                print("step:", i)

            for j in range(len(h_seq) - 1):
                dz = z_values[j] - z_values[j + 1]
                h = h_seq[j + 1]
                rec = self.solver.solve(recovered_h, dz)
                phase = np.angle(rec)
                recovered_h = h * np.exp(1j * phase)

            lidx = len(h_seq) - 1
            for j in range(len(h_seq) - 1):
                dz = z_values[lidx - j] - z_values[lidx - j - 1]
                h = h_seq[lidx - j - 1]
                rec = self.solver.solve(recovered_h, dz)
                phase = np.angle(rec)
                recovered_h = h * np.exp(1j * phase)

        recovered_h = self.solver.solve(recovered_h, z_values[0])
        return recovered_h
