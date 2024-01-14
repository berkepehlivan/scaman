import numpy as np


class NumpyEigensolver:
    """
    A class for computing the eigenvalues and eigenvectors of a matrix using NumPy.

    Parameters:
    -----------
    solver : str
        The method used to compute the eigenvalues and eigenvectors. 
        Can be either "dense" for dense matrices or "arpack" for sparse symmetric matrices.
    check_finite : bool, optional (default=True)
        Whether to check that the input matrix contains only finite values.
    
    Attributes:
    -----------
    eigenvectors_ : array-like, shape (n_components, n_features)
        The eigenvectors corresponding to the eigenvalues computed by the solver.
    eigenvalues_ : array-like, shape (n_components,)
        The eigenvalues computed by the solver.

    Methods:
    --------
    fit(X, num_components=None)
        Fit the solver to the input matrix X, computing the specified number of eigenvectors and eigenvalues.
    
    """

    def __init__(self, solver="dense", check_finite=True):
        self.solver = solver
        self.check_finite = check_finite
        self.eigenvalues_ = None
        self.eigenvectors_ = None

    def fit(self, X, num_components=None):
        """
        Fit the solver to the input matrix X, computing the specified number of eigenvectors and eigenvalues.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input matrix to compute eigenvectors and eigenvalues on.
        num_components : int or None, optional (default=None)
            The number of components (eigenvectors and eigenvalues) to compute. 
            If None, compute all components.
        
        Returns:
        --------
        eigenvalues : array-like, shape (n_components,)
            The eigenvalues computed by the solver.
        eigenvectors : array-like, shape (n_features, n_components)
            The eigenvectors corresponding to the eigenvalues computed by the solver.
        """
        self.X = X
        if self.check_finite and not np.isfinite(X).all():
            raise ValueError("Input matrix must contain only finite values")

        if self.solver == "dense":
            eigenvalues, eigenvectors = np.linalg.eigh(X)
        #if self.solver =='sparse':
            eigenvalues, eigenvectors = np.linalg.eigh(X)
        elif self.solver == "sparse":
            from scipy.sparse.linalg import eigsh
            eigenvalues, eigenvectors = eigsh(X, k=num_components, which="LM")


        if num_components is not None:
            eigenvalues = eigenvalues[:num_components]
            eigenvectors = eigenvectors[:, :num_components]

        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors

        return eigenvalues, eigenvectors
