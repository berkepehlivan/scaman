import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from eigensolvers.numpy_eigensolver import NumpyEigensolver
from eigensolvers.scipy_eigensolver import ScipyEigensolver
from eigensolvers.slepc_eigensolver import SlepcEigensolver
from sklearn.neighbors import NearestNeighbors

class LE:
    def __init__(self, data, n_components=2, k=5, sigma=1,solver='numpy',
                 slepc_solver='KRYLOVSCHUR',affinity='networkx',normalize=False):
        self.data = data
        self.n_components = n_components
        self.k = k
        self.sigma = sigma
        self.solver = solver
        self.slepc_solver = slepc_solver
        self.normalize = normalize
        self.affinity=affinity
    
    def _compute_affinity_matrix(self):
        """
        Computes the affinity matrix (W) for a given dataset X using a Gaussian kernel.

        Parameters:
        X (numpy array): Input dataset of shape (n_samples, n_features).
        k (int): Number of nearest neighbors to consider for each data point.
        sigma (float): Width of the Gaussian kernel.

        Returns:
        W (numpy array): Affinity matrix of shape (n_samples, n_samples).
        """
        n_samples = self.data.shape[0]
        W = np.zeros((n_samples, n_samples))
        np.fill_diagonal(W, 1.0)
        for i in range(n_samples):
            distances = np.linalg.norm(self.data - self.data[i], axis=1)
            nearest_indices = np.argsort(distances)[1:self.k+1]
            for j in nearest_indices:
                W[i][j] = np.exp(-distances[j]**2 / (2*self.sigma**2)) # Gaussian kernel
                W[j][i] = W[i][j]
        return W
    
    def _compute_affinity_matrix_networkx(self):
        G = nx.Graph()
        num_samples = self.data.shape[0]
        W = np.zeros((num_samples, num_samples))
        np.fill_diagonal(W, 1.0)
        
        for i in range(num_samples):
            distances = np.linalg.norm(self.data - self.data[i], axis=1)
            nearest_indices = np.argsort(distances)[1:self.k+1]
            
            for j in nearest_indices:
                G.add_edge(i, j, weight=distances[j])
        
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                if G.has_edge(i, j):
                    dist = G[i][j]['weight']
                    affinity = np.exp(-dist**2 / (2*self.sigma**2))
                    W[i, j] = affinity
                    W[j, i] = affinity
        
        return W

    def _compute_affinity_matrix_flann(self):
        import pyflann
        num_points = self.data.shape[0]
        num_neighbors= self.k
        if num_neighbors is None:
            num_neighbors = num_points - 1

        # Initialize FLANN with the data
        flann = pyflann.FLANN()
        params = flann.build_index(self.data, algorithm="kmeans", branching=32, iterations=7, checks=16) ## target precision autotune ile ve kdtree testleri yap.

        # Perform nearest neighbor search
        neighbors, distances = flann.nn_index(self.data, num_neighbors + 1, checks=params["checks"])

        # Remove the first column of distances (self-distances)
        #distances = distances[:, 1:]

        affinity_matrix = np.zeros((num_points, num_points))
        np.fill_diagonal(affinity_matrix, 1.0)

        # Construct affinity matrix based on distances
        for i in range(num_points):
            for j in range(num_neighbors):
                affinity_matrix[i, neighbors[i, j + 1]] = np.exp(-distances[i, j] ** 2 / (2 * self.sigma ** 2))
                affinity_matrix[neighbors[i, j + 1], i] = np.exp(-distances[i, j] ** 2 / (2 * self.sigma ** 2))
        
        W = affinity_matrix
        return W
    
    def _compute_unnormalized_laplacian_matrix(self, W):
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        return L,D
    
    def _compute_normalized_laplacian_matrix(self, W):
        D = np.diag(np.sum(W, axis=1))
        D_inv_sqrt = np.diag(1 / np.sqrt(np.sum(W, axis=1)))
        L = D_inv_sqrt @ (D - W) @ D_inv_sqrt
        return L, D
    
    def _compute_embedding(self,L,D):
        if self.solver == "numpy":
            eigenvalues, eigenvectors = NumpyEigensolver().fit(L, num_components=self.n_components+1)
           
        elif self.solver == "scipy":
            eigenvalues, eigenvectors = ScipyEigensolver().fit(L, num_components=self.n_components+1)
            
        elif self.solver == "slepc":
            eigenvalues, eigenvectors = SlepcEigensolver(L, solver=self.slepc_solver, nev=self.n_components,magnitude="smallest",
                                                         tol=1e-15,max_it=10000)
            eigenvectors = eigenvectors[:,:self.n_components+1]
        else:
            raise Exception("Invalid solver. Must be one of 'numpy', 'scipy', or 'slepc'")
        
        #eigenvalues, eigenvectors = eigsh(L, k=self.n_components+1, which='SM')
        D_I = np.sqrt(D.diagonal()[np.newaxis, :self.n_components]) ## özvektörleri hesaplamadan önce bu çarpımı yap
        Y = eigenvectors[:, 1:] * D_I[:, 1:]
        return Y
    
    def fit_transform(self):
        
        def check_affinity(self):
            if self.affinity=='networkx':
                W = self._compute_affinity_matrix_networkx()
                return W
            elif self.affinity=='numpy':
                W = self._compute_affinity_matrix()
                return W
            elif self.affinity=='flann':
                W = self._compute_affinity_matrix_flann()
                return W
            else:
                print('affinity value should either be networkx or flann.')
        W = check_affinity(self)  
        if self.normalize:
            L,D = self._compute_normalized_laplacian_matrix(W)
        else:
            L,D = self._compute_unnormalized_laplacian_matrix(W)
        Y = self._compute_embedding(L,D)
        return Y
    
    def plot_2d(self,Y,labels=None):
        if labels is not None:
            plt.figure(figsize=(10, 10))
            plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='viridis')
            plt.colorbar()
            plt.show() 
        else:
            labels = np.random.randint(0, 2, size=Y.shape[0])
            cmap = cm.get_cmap('viridis', np.max(labels) - np.min(labels) + 1)
            plt.figure(figsize=(10, 10))
            plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap=cmap)
            plt.colorbar()
            plt.show()
    
    def plot_3d(self,Y,labels):
        if labels is not None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=labels, cmap='viridis')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
        else:
            labels = np.random.randint(0, 2, size=Y.shape[0])
            cmap = cm.get_cmap('viridis', np.max(labels) - np.min(labels) + 1)
            plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=labels, cmap='viridis')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()


            