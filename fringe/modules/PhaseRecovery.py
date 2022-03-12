
class MultiDistancePhaseOptimizer:
    def __init__(self, solver):
        """
        Recovers the phase information by multiple intensity-only images captured from different distances.

        Parameters
        ----------
        solver : class(solvers.base.Solver)
            The propagation solver class (e.g., AngularSpectrum) which encapsulates Solver.solve(input_, *args) to
            apply propagation algorithm on the input field.
        """
        self.solver = solver
        self.backend = solver.backend

    def optimize(self, image_seq, k, z_values, iterations=20):
        """
        Optimizes phase by Gerchberg–Saxton algorithm using multiple diversified acquisitions.

        Parameters
        ----------
        image_seq: array_like, list, ndarray - dtype: float32
            Sequence of intensity-only images.

        k:  float
            Wave number : 2πn/λ

        z_values:   array-like, list
            Axial sample-to-sensor distances of which each hologram was acquired from.

        iterations: int
            Number of iterations for optimization.
        """
        recovered_field = image_seq[0]

        for i in range(iterations):
            if i % 5 == 0:
                print("step:", i)

            for j in range(len(image_seq) - 1):
                dz = z_values[j] - z_values[j + 1]
                img = image_seq[j + 1]
                rec = self.solver.solve(recovered_field, k, dz)
                phase = self.backend.exp(self.backend.complex(
                    real=self.backend.zeros_like(rec, dtype='float32'),
                    imag=self.backend.angle(rec)))
                recovered_field = self.backend.multiply(img, phase)

            last_i = len(image_seq) - 1
            for j in range(len(image_seq) - 1):
                dz = z_values[last_i - j] - z_values[last_i - j - 1]
                img = image_seq[last_i - j - 1]
                rec = self.solver.solve(recovered_field, k, dz)
                phase = self.backend.exp(self.backend.complex(
                    real=self.backend.zeros_like(rec, dtype='float32'),
                    imag=self.backend.angle(rec)))
                recovered_field = self.backend.multiply(img, phase)

        return self.solver.solve(recovered_field, k, z_values[0])
