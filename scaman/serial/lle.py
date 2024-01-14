import matplotlib.pyplot as plt
import numpy as np
import pyflann

from eigensolvers.numpy_eigensolver import NumpyEigensolver
from eigensolvers.scipy_eigensolver import ScipyEigensolver
from eigensolvers.slepc_eigensolver import SlepcEigensolver
from sklearn.neighbors import NearestNeighbors

class LLE:
    def __init__(self, n_components=2, n_neighbors=5, neighbors='sklearn' ,solver="slepc", 
                 slepc_solver='ARNOLDI',tol=1e-3,max_it=100):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.solver = solver
        self.slepc_solver = slepc_solver
        self.reg = 1e-3
        self.tol = tol
        self.max_it = max_it
        self.neighbors = neighbors
        

    def fit_transform(self, data):
        self.data = data

        # Step 1: Find nearest neighbors

        if self.neighbors == 'sklearn':
            neighbors, indices, distances = self._find_neighbors(data)
        elif self.neighbors == 'flann':
            neighbors, indices, distances = self._find_neighbors_flann(data)
        else:
            raise Exception("Invalid neighbors method. Must be one of 'sklearn' or 'flann'")

        # Step 2: Compute weights
        weights = self._compute_weights(indices)

        # Step 3: Compute embedding
        embedding = self._compute_embedding(weights)

        return embedding
    
    def plot(self, embedding, labels=None):
        # Plot the embedding with color mapping
        if labels is not None:
            plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis')
            plt.colorbar()
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1])
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'LLE Embedding (n_components={self.n_components}, n_neighbors={self.n_neighbors}, solver={self.solver})')
        plt.show()
    
    def _find_neighbors(self, data):
        # Compute nearest neighbors using the sklearn library
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1, algorithm='ball_tree').fit(data)
        distances,indices = nbrs.kneighbors(data, return_distance=True) 
        neighbors = indices[:, 1:]
        indices = neighbors.astype(int)
        return neighbors, indices, distances
    
    def _find_neighbors_flann(self, data):
        # Compute nearest neighbors using the PyFlann library
        flann = pyflann.FLANN()
        params = flann.build_index(data, algorithm='kdtree',trees=4)
        distances, indices = flann.nn_index(data, self.n_neighbors+1, checks=params["checks"])
        neighbors = indices[:, 1:]
        indices = neighbors.astype(int)
        return neighbors, indices, distances

    def _compute_weights(self, indices):
        # Compute the weights for each data point
        N = self.data.shape[0]
        weights = np.zeros((N, N))
        for i in range(N):
            # Step 1: Compute the local covariance matrix
            X = self.data[indices[i,:],:] - self.data[i,:]
            C = X @ X.T

            # Step 2: Compute the weights using the regularization method
            C = np.add(C, self.reg * np.eye(indices.shape[1]).astype(float))
            inv = np.linalg.inv(C)
            w = np.ones(indices.shape[1]) @ inv
            w /= np.sum(w)

            weights[i,indices[i,:]] = w

        print('Weights Shape:\n',weights.shape)
        
        return weights
        
    def _compute_embedding(self, weights):
        # Compute the matrix M
        I = np.eye(weights.shape[0])
        W = weights
        M = (I - W).T @ (I - W)

        # Solve for the weights
        if self.solver == "numpy":
            eigenvalues, eigenvectors = NumpyEigensolver().fit(M, num_components=self.n_components)
        elif self.solver == "scipy":
            eigenvalues, eigenvectors = ScipyEigensolver().fit(M, num_components=self.n_components)
        elif self.solver == "slepc":
            eigenvalues, eigenvectors = SlepcEigensolver(M, solver=self.slepc_solver, nev=self.n_components,magnitude="smallest",tol=self.tol,max_it=self.max_it)
        else:
            raise Exception("Invalid solver. Must be one of 'numpy', 'scipy', or 'slepc'")

        if eigenvectors.shape[1] > self.n_components:
        # Select the first n_components non-zero eigenvectors
            eigenvectors = eigenvectors[:,0:self.n_components]
            
        # Compute the embeddings
        inv_weights_sum = np.sqrt(1 / np.sum(weights, axis=1))
        embeddings = eigenvectors * inv_weights_sum[:, np.newaxis]

        return embeddings
    

