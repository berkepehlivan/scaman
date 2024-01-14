import scipy.linalg as la
import numpy as np

class ScipyEigensolver(object):
    """
    A wrapper class for the eigensolver in scipy.linalg.eigh.

    Parameters:
    -----------
    solver : string, optional (default="dense")
        The solver to use. Options are "dense" and "arpack".
    check_finite : bool, optional (default=True)
        Whether to check that the input matrix contains only finite values.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Attributes:
    -----------
    eigenvalues_ : array-like, shape (n_components,)
        The eigenvalues computed by the solver.
    eigenvectors_ : array-like, shape (n_components, n_features)
        The eigenvectors corresponding to the eigenvalues computed by the solver.

    Methods:
    --------
    fit(X, num_components=None)
        Fit the solver to the input matrix X, computing the specified number of eigenvectors and eigenvalues.

    """

    def __init__(self, solver="arpack", check_finite=True):
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
        #if self.check_finite and not np.isfinite(X).all():
        #   raise ValueError("Input matrix must contain only finite values")

        if self.solver == "dense":
            eigenvalues, eigenvectors = la.eigh(X)
        elif self.solver == "arpack":
            from scipy.sparse.linalg import eigsh
            eigenvalues, eigenvectors = eigsh(X, k=num_components, which="SM")

        if num_components is not None:
            eigenvalues = eigenvalues[:num_components]
            eigenvectors = eigenvectors[:, :num_components]
        elif num_components is None:
            eigenvalues = eigenvalues
            eigenvectors = eigenvectors
        elif num_components > eigenvalues.shape[0]:
            eigenvalues = eigenvalues
            eigenvectors = eigenvectors
        else:
            raise ValueError("num_components must be less than or equal to the number of eigenvalues")
        
        return eigenvalues, eigenvectors
    

