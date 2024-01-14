import matplotlib.pyplot as plt
import numpy as np
import os
import time

from eigensolvers.numpy_eigensolver import NumpyEigensolver
from eigensolvers.scipy_eigensolver import ScipyEigensolver
from eigensolvers.slepc_eigensolver import SlepcEigensolver

EIGENSOLVERS = {
    'scipy': ScipyEigensolver,
    'numpy': NumpyEigensolver,
    'slepc': SlepcEigensolver
}

class PCA:
    def __init__(self, n_components, eig_method='slepc',slepc_solver='KRYLOVSCHUR',tol=1e-3,max_it=100):
        self.n_components = n_components
        self.eig_method = eig_method
        self.mean_ = None
        self.components_ = None
        self.slepc_solver = slepc_solver
        self.tol = tol
        self.max_it = max_it

    def _center_data(self, X):
        return X - np.mean(X, axis=0)

    def fit_transform(self, X):
        X_centered = self._center_data(X)
        n = X.shape[0]

        
        covariance_matrix = np.dot(X_centered.T, X_centered) / (n - 1)
        t0 = time.time()
        solver_class = EIGENSOLVERS.get(self.eig_method)
        if solver_class is None:
            raise ValueError(
                "Invalid eig_method. Valid options are 'scipy', 'numpy', and 'slepc'.")
        if solver_class == SlepcEigensolver:
            eigenvalues,eigenvectors = SlepcEigensolver(data=covariance_matrix,solver=self.slepc_solver, nev=self.n_components,tol=self.tol,max_it=self.max_it)
        else:
            solver = solver_class()
            eigenvalues, eigenvectors = solver.fit(
                covariance_matrix, num_components=self.n_components)
            
        idx = np.argsort(eigenvalues)[::-1][:self.n_components]
        eigenvectors = eigenvectors[:, idx]

        self.mean_ = np.mean(X, axis=0)
        self.components_ = eigenvectors

        t1 = time.time()
        print(f'Time taken to compute eigenvalues: {t1 - t0:.4f} seconds')

        return np.dot(X_centered, eigenvectors)

    def transform(self, X):
        X_centered = self._center_data(X)
        return np.dot(X_centered, self.components_)

    def inverse_transform(self, X):
        return np.dot(X, self.components_.T) + self.mean_

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
        #if not os.path.exists(output_dir):
        #    os.makedirs(output_dir)

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
                #plt.savefig(
                #    f'{output_dir}/eigenvector_{i+1}_pca_plot_{self.eig_method}.png', dpi=300)
                plt.show()
