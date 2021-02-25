import numpy as np
import tensorflow as tf


class AngularSpectrumSolver:
    def __init__(self, shape, dx, dy, wavelength, dtype_c=tf.complex64, dtype_f=tf.float32):
        self.shape = shape
        self.dx = dx
        self.dy = dy
        self.k = 2 * np.pi / wavelength
        self.k2 = self.k * self.k
        kx = np.fft.fftfreq(shape[0], dx / (2 * np.pi))
        ky = np.fft.fftfreq(shape[1], dy / (2 * np.pi))
        kx, ky = np.meshgrid(kx, ky, indexing='ij', sparse=True)
        self.kz2 = tf.convert_to_tensor(kx * kx + ky * ky, dtype=dtype_f)
        self.valid_mask = self.k2 > self.kz2

        self.dtype_f = dtype_f
        self.dtype_c = dtype_c

    def propagator(self, z):
        p = tf.zeros(self.shape, dtype=self.dtype_f)
        p = tf.complex(real=p, imag=tf.math.sqrt(self.k2 - self.kz2) * z)
        return tf.math.exp(p)

    def solve(self, initializer, z):
        prop = self.propagator(z)
        return tf.signal.ifft2d(prop * tf.signal.fft2d(initializer))


class MultiHeightPhaseRecovery:
    def __init__(self, solver):
        self.solver = solver

    def solve(self, h_seq, z_values, iterations=20):
        recovered_h = h_seq[0]

        for i in range(iterations):
            if i % 5 == 0:
                print("step:", i)

            for j in range(len(h_seq) - 1):
                dz = z_values[j] - z_values[j + 1]
                h = h_seq[j + 1]
                rec = self.solver.solve(recovered_h, dz)
                phase = tf.exp(tf.complex(real=tf.zeros_like(rec, dtype=self.solver.dtype_f),
                                          imag=tf.math.angle(rec)))
                recovered_h = tf.multiply(h, phase)

            last_i = len(h_seq) - 1
            for j in range(len(h_seq) - 1):
                dz = z_values[last_i - j] - z_values[last_i - j - 1]
                h = h_seq[last_i - j - 1]
                rec = self.solver.solve(recovered_h, dz)
                phase = tf.exp(tf.complex(real=tf.zeros_like(rec, dtype=self.solver.dtype_f),
                                          imag=tf.math.angle(rec)))
                recovered_h = tf.multiply(h, phase)

        recovered_h = self.solver.solve(recovered_h, z_values[0])
        return recovered_h
