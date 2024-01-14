import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('/Users/berke/Codes/scaman/scaman')
from eigensolvers.scipy_eigensolver import ScipyEigensolver
from eigensolvers.numpy_eigensolver import NumpyEigensolver
from eigensolvers.slepc_eigensolver_parallel import SlepcParallelEigensolver
import os
from mpi4py import MPI


EIGENSOLVERS = {
    'scipy': ScipyEigensolver,
    'numpy': NumpyEigensolver,
    'slepc': SlepcParallelEigensolver
}


class PCA:
    def __init__(self, n_components, eig_method='scipy'):
        self.n_components = n_components
        self.eig_method = eig_method

    def _center_data(self, X):
        return X - np.mean(X, axis=0)

    def fit_transform(self, X):
        ###-----MPI Profiler for Debugging-----###
        start_time = MPI.Wtime()
        ###----------###
        X_centered = self._center_data(X)
        n = X.shape[0]

        t0 = time.time()
        covariance_matrix = np.dot(X_centered.T, X_centered) / (n - 1)


        solver_class = EIGENSOLVERS.get(self.eig_method)
        if solver_class is None:
            raise ValueError(
                "Invalid eig_method. Valid options are 'scipy', 'numpy', and 'slepc'.")

        if solver_class == SlepcParallelEigensolver:
            eigenvalues,eigenvectors = SlepcParallelEigensolver(data=covariance_matrix,solver='KRYLOVSCHUR', nev=self.n_components) # type: ignore
        else:
            solver = solver_class()
            eigenvalues, eigenvectors = solver.fit(
                covariance_matrix, num_components=self.n_components)

        idx = np.argsort(eigenvalues)[::-1][:self.n_components]
        eigenvectors = eigenvectors[:, idx]


        t1 = time.time()
        print(f'Time taken to compute eigenvalues: {t1 - t0:.4f} seconds')
        ###-----MPI Profiler for Debugging-----###
        elapsed_time = MPI.Wtime() - start_time
        print("Elapsed time for fit_transform: {:.4f} seconds".format(
            elapsed_time))
        ###----------###
        return np.dot(X_centered, eigenvectors)

    def plot(self, X, eigenvectors=None, color=None, output_dir="output"):
        """
        Plots the first three eigenvectors of the PCA algorithm separately and saves the output as PNG files.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        eigenvectors : array-like, shape (n_samples, n_components), optional
            The eigenvectors of the PCA algorithm. If not provided, they are computed using fit_transform.
        color : array-like, shape (n_samples,), optional
            The color map for the eigenvectors. If not provided, the default color map is used.
        output_dir : str, optional
            The directory to save the output plots. Defaults to "output".
        """

        if eigenvectors is None:
            eigenvectors = self.fit_transform(X)

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range(min(3, self.n_components)):
            fig, ax = plt.subplots(figsize=(10, 7))

            if color is not None:
                cmap = plt.get_cmap('viridis')  # type: ignore
                norm = plt.Normalize(  # type: ignore
                    vmin=min(color), vmax=max(color))  # type: ignore
                scatter = ax.scatter(
                    range(len(eigenvectors)), eigenvectors[:, i], c=color, cmap=cmap, norm=norm)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.ax.set_ylabel('Color map')
            else:
                ax.scatter(range(len(eigenvectors)), eigenvectors[:, i])

            ax.set_title(f'Principal Component {i+1}')
            plt.savefig(
                f'{output_dir}/eigenvector_{i+1}_pca_plot_{self.eig_method}.png', dpi=300)
            plt.show()
