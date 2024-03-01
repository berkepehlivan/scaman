import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors

from eigensolvers.numpy_eigensolver import NumpyEigensolver
from eigensolvers.scipy_eigensolver import ScipyEigensolver
from eigensolvers.slepc_eigensolver import SlepcEigensolver


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

        # Compute the pairwise distances
        pairwise_distances = squareform(pdist(self.data))

        # Get the indices of the k nearest neighbors for each point
        nearest_indices = np.argsort(pairwise_distances, axis=1)[:, 1:self.k+1]

        # Compute the Gaussian kernel for each pair of points
        W = np.exp(-pairwise_distances**2 / (2*self.sigma**2))

        # Create an affinity matrix with zeros
        affinity_matrix = np.zeros_like(W)

        # Fill the affinity matrix with the values from the Gaussian kernel for the k nearest neighbors
        for i in tqdm(range(self.data.shape[0]), desc="Computing affinity matrix with numpy and sklearn"):
            affinity_matrix[i, nearest_indices[i]] = W[i, nearest_indices[i]]
            affinity_matrix[nearest_indices[i], i] = W[nearest_indices[i], i]
        #print('affinity matrix constructed...')
        return affinity_matrix

    def _compute_affinity_matrix_networkx(self):
        num_samples = self.data.shape[0]
        W = np.zeros((num_samples, num_samples))
        np.fill_diagonal(W, 1.0)

        # Compute all pairwise distances
        distances = cdist(self.data, self.data)

        # Create a graph
        G = nx.from_numpy_array(distances)
        print('Graph created...')
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        print('Number of nodes:',num_nodes)
        print('Number of edges:',num_edges)

        # Compute nearest neighbors for each point
        nearest_indices = np.argsort(distances, axis=1)[:, 1:self.k+1]

        # Add edges to the graph for k nearest neighbors
        for i in tqdm(range(num_samples), desc="Adding edges and computing affinity with networkx"):
            for j in nearest_indices[i]:
                G.add_edge(i, j, weight=distances[i, j])

        # Compute affinity
        for i, j in tqdm(G.edges, desc="Computing affinity with networkx"):
            dist = G[i][j]['weight']
            affinity = np.exp(-dist**2 / (2*self.sigma**2))
            W[i, j] = affinity
            W[j, i] = affinity
        print('affinity matrix constructed...')
        print('W shape:',W.shape)
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
        for i in tqdm(range(num_points), desc="Computing affinity matrix with FLANN"):
            for j in range(num_neighbors):
                affinity_matrix[i, neighbors[i, j + 1]] = np.exp(-distances[i, j] ** 2 / (2 * self.sigma ** 2))
                affinity_matrix[neighbors[i, j + 1], i] = np.exp(-distances[i, j] ** 2 / (2 * self.sigma ** 2))
        
        W = affinity_matrix
        return W
    


    def _compute_affinity_matrix_parallel(self):
        # Compute the pairwise distances
        pairwise_distances = squareform(pdist(self.data))

        # Get the indices of the k nearest neighbors for each point
        nearest_indices = np.argsort(pairwise_distances, axis=1)[:, 1:self.k+1]

        # Compute the Gaussian kernel for each pair of points
        W = np.exp(-pairwise_distances**2 / (2*self.sigma**2))

        # Create an affinity matrix with zeros
        affinity_matrix = np.zeros_like(W)

        # Fill the affinity matrix with the values from the Gaussian kernel for the k nearest neighbors
        affinity_matrices = Parallel(n_jobs=-1)(delayed(compute_affinity)(i, nearest_indices, W) for i in tqdm(range(self.data.shape[0]), desc="Computing affinity matrix in parallel"))

        # Sum all the affinity matrices to get the final affinity matrix
        affinity_matrix = np.sum(affinity_matrices, axis=0)
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
            eigenvalues, eigenvectors = NumpyEigensolver().fit(L, num_components=self.n_components)
           
        elif self.solver == "scipy":
            scipy_solver = ScipyEigensolver(solver='arpack')
            eigenvalues, eigenvectors = scipy_solver.fit(L, num_components=self.n_components)
            
        elif self.solver == "slepc":
            eigenvalues, eigenvectors = SlepcEigensolver(L, solver=self.slepc_solver, nev=self.n_components,magnitude="smallest",
                                                         tol=1e-15,max_it=10000)
            eigenvectors = eigenvectors[:,:self.n_components]
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
            elif self.affinity=='parallel':
                W = self._compute_affinity_matrix_parallel()
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


            
