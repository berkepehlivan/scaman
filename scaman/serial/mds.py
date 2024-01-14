import matplotlib.pyplot as plt
import numpy as np
import os
import time

from eigensolvers.numpy_eigensolver import NumpyEigensolver
from eigensolvers.scipy_eigensolver import ScipyEigensolver
from eigensolvers.slepc_eigensolver import SlepcEigensolver
from scipy.spatial.distance import cdist
from sklearn.datasets import make_swiss_roll

class MDS:
    def __init__(self ,n_components, eig_method='scipy',slepc_solver='KRYLOVSCHUR'):
        self.n_components = n_components
        self.eig_method = eig_method
        self.slepc_solver = slepc_solver

    def _pairwise_distances(self, X):
        return cdist(X, X)

    def _compute_distances(self, X):
        return cdist(X, X)

    def _compute_stress(self, distances, distances_computed, n):
        delta = distances[np.triu_indices(n, k=1)]
        distances_computed = distances_computed[np.triu_indices(n, k=1)]
        stress = np.sqrt(np.sum((distances_computed - delta)
                         ** 2)) / np.sqrt(np.sum(delta ** 2))
        return stress

    def fit_transform(self, X):
        

        EIGENSOLVERS = {
            'scipy': ScipyEigensolver,
            'numpy': NumpyEigensolver,
            'slepc': SlepcEigensolver
        }

        distances = self._pairwise_distances(X)
        n = distances.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        distances = -0.5 * H.dot(distances ** 2).dot(H)

        solver_class = EIGENSOLVERS.get(self.eig_method)
        if solver_class == SlepcEigensolver:
            eigenvalues,eigenvectors = SlepcEigensolver(data=distances, solver=self.slepc_solver, nev=self.n_components,tol=1e-9,max_it=1000)
        
        elif solver_class == NumpyEigensolver:
            numpy_solver = NumpyEigensolver(solver='dense', check_finite=False)
            eigenvalues, eigenvectors = numpy_solver.fit(X=distances, num_components=self.n_components)
            idx = np.argsort(eigenvalues)[::-1][:self.n_components]
            eigenvectors = eigenvectors[:, idx]

        elif solver_class == ScipyEigensolver:
            scipy_solver = ScipyEigensolver(solver='arpack', check_finite=False)
            eigenvalues, eigenvectors = scipy_solver.fit(X=distances, num_components=self.n_components)
            idx = np.argsort(eigenvalues)[::-1][:self.n_components]
            eigenvectors = eigenvectors[:, idx]
        else:
            raise ValueError(
                "Invalid eig_method. Valid options are 'scipy', 'numpy', and 'slepc'.")

        #solver = solver_class()
        #eigenvalues, eigenvectors = solver.fit(
        #    distances, num_components=self.n_components)
       

        #stress = self._compute_stress(
        #   distances, self._compute_distances(eigenvectors), n)
        #print(f'Stress value: {stress:.4f}')

        return eigenvectors

    def plot(self, X, eigenvectors=None, color=None, output_dir="output"):
        """
        Plots the first three eigenvectors of the MDS algorithm separately and saves the output as PNG files.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        eigenvectors : array-like, shape (n_samples, n_components), optional
            The eigenvectors of the MDS algorithm. If not provided, they are computed using fit_transform.
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
                cmap = plt.get_cmap('viridis') # type: ignore
                norm = plt.Normalize(vmin=min(color), vmax=max(color)) # type: ignore
                scatter = ax.scatter(
                    range(len(eigenvectors)), eigenvectors[:, i], c=color, cmap=cmap, norm=norm)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.ax.set_ylabel('Color map')

                
            else:
                ax.scatter(range(len(eigenvectors)), eigenvectors[:, i])

            ax.set_title(f'Eigenvector {i+1} (MDS - {self.eig_method})')
            ax.set_xlabel('Data point index')
            ax.set_ylabel(f'Eigenvector {i+1}')

            output_file = os.path.join(
                output_dir, f'eigenvector_{i+1}_mds_{self.eig_method}.png')
            plt.savefig(output_file, dpi=300)
            plt.show()
