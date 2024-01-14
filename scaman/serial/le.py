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
                 slepc_solver='KRYLOVSCHUR',normalize=False):
        self.data = data
        self.n_components = n_components
        self.k = k
        self.sigma = sigma
        self.solver = solver
        self.slepc_solver = slepc_solver
        self.normalize = normalize
    
    def _compute_affinity_matrix(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.data.shape[0]))
        for i in range(self.data.shape[0]):
            distances = np.linalg.norm(self.data - self.data[i], axis=1)
            nearest_indices = np.argsort(distances)[1:self.k+1]
            for j in nearest_indices:
                G.add_edge(i, j, weight=distances[j])
        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G))
        W = np.zeros((self.data.shape[0], self.data.shape[0]))
        for i in range(self.data.shape[0]):
            for j in range(i+1, self.data.shape[0]):
                if j in shortest_paths[i]:
                    dist = shortest_paths[i][j]
                    W[i][j] = np.exp(-dist**2 / (2*self.sigma**2)) # Gaussian kernel
                    W[j][i] = W[i][j]
        return W
    
    def _compute_approximate_affinity(self):
        pass
    

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
                                                         tol=1e-9,max_it=10000)
            eigenvectors = eigenvectors[:,:self.n_components+1]
        else:
            raise Exception("Invalid solver. Must be one of 'numpy', 'scipy', or 'slepc'")
        
        #eigenvalues, eigenvectors = eigsh(L, k=self.n_components+1, which='SM')
        if self.normalize:
            Y = eigenvectors
        else:
            D_I = np.sqrt(D.diagonal()[np.newaxis, :self.n_components+1])
            Y = eigenvectors[:, 1:] * D_I[:, 1:]
        return Y
    
    def fit_transform(self):
        W = self._compute_affinity_matrix()
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


            