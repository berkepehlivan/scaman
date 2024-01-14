import numpy as np
from scipy.linalg import eigh

class FEAST_output_struct:
    def __init__(self, ei, ee, iterCount, complexCounter, dimensionReduceCounter, itError, itIndex, isConverged):
        self.ei = ei
        self.ee = ee
        self.iterCount = iterCount
        self.complexCounter = complexCounter
        self.dimensionReduceCounter = dimensionReduceCounter
        self.itError = itError
        self.itIndex = itIndex
        self.isConverged = isConverged

def gausslegendre(Ne):
    xe, we = np.polynomial.legendre.leggauss(Ne)
    return xe, we

def FeastEigensolver(A, B, M0, Y, Ne, lambdamin, lambdamax, tol, maxIter, options, aei):
    '''    Implements the FEAST algorithm for solving eigenvalue problems.

    Parameters
    ----------
    A : ndarray
        The matrix for which to solve the eigenvalue problem.
    B : ndarray
        The matrix defining the eigenvalue problem.
    M0 : int
        The initial guess for the number of eigenvalues in the interval. (estDim*1,5)
    Y : ndarray
        The initial guess for the eigenvectors. (np.rand ile oluştur shape kadar)
    Ne : int
        The number of quadrature points.(8)
    lambdamin : float
        The lower bound of the interval in which to search for eigenvalues.(0 ya da en küçük özdeğerden küçük 0'dan büyük)
    lambdamax : float
        The upper bound of the interval in which to search for eigenvalues.(estDim +1 ile ikisinin ortalaması)
    tol : float
        The tolerance for convergence. (1e-6)
    maxIter : int
        The maximum number of iterations. (100)
    options : int
        The option for the subspace iteration method (0, 1, 2, or 3). (1)
    aei : list
        The list of actual eigenvalues for error calculation.

    Returns
    -------
    FEAST_output_struct
        The output structure containing the eigenvalues, eigenvectors, and other information about the iterations.
    '''
    N = A.shape[0]
    r = (lambdamax - lambdamin) / 2
    xe, we = gausslegendre(Ne)

    it_error = []
    it_ind = []
    X_m = np.empty((N, 0), dtype=float)
    complexCounter = 0
    dimensionReduceCounter = 0
    Q = np.zeros_like(Y)

    for i in range(1, maxIter + 1):
        lambda_m = []
        for e in range(Ne):
            teta = -(np.pi / 2) * (xe[e] - 1)
            z = (lambdamax + lambdamin) / 2 + r * np.exp(1j * teta)
            Qe = np.linalg.solve(z * B - A, Y) #burası paralelleşecek
            Q = Q-(we[e] / 2) * np.real(r * np.exp(1j * teta) * Qe)

        if options == 0:
            Aq = Q.T @ A @ Q
            Bq = Q.T @ B @ Q
            # Add a small positive number to the diagonal elements of Bq
            #Bq = Bq + np.eye(Bq.shape[0]) * 1e-6
        elif options == 1:
            Q, R = np.linalg.qr(Q)
            Aq = Q.T @ A @ Q
            Bq = Q.T @ B @ Q
        elif options == 2:
            Aq = Q.T @ A @ Q
            Bq = Q.T @ B @ Q
            rB = np.linalg.matrix_rank(Bq)
            if rB != M0 and rB >= X_m.shape[1]:
                M0 = rB
                Q = Q[:, :rB]
                Y = Y[:, :rB]
                Aq = Q.T @ A @ Q
                Bq = Q.T @ B @ Q
                dimensionReduceCounter += 1
        elif options == 3:
            Q, R = np.linalg.qr(Q)
            Aq = Q.T @ A @ Q
            Bq = Q.T @ B @ Q
            rB = np.linalg.matrix_rank(Bq)
            if rB != M0 and rB >= X_m.shape[1]:
                M0 = rB
                Q = Q[:, :rB]
                Y = Y[:, :rB]
                Aq = Q.T @ A @ Q
                Bq = Q.T @ B @ Q
                dimensionReduceCounter += 1

        ritz, phi = eigh(Aq, Bq)

        X_n = Q @ phi
        for j in range(M0):
            if lambdamin <= ritz[j] <= lambdamax:
                lambda_m.append(ritz[j])
                #lambda_m[cnt]=ritz[j]
                X_m = np.column_stack((X_m, X_n[:, j]))
                #cnt=cnt+1

        if len(lambda_m) == len(aei):
            Flag = 1
            max_error = 0
            max_error_ind = -1
            for k in range(len(lambda_m)):
                x = X_m[:, k]
                x = x / np.linalg.norm(x)
                current_error = np.linalg.norm(A @ x - lambda_m[k] * B @ x)
                if current_error > tol:
                    Flag = 0
                if current_error > max_error:
                    max_error = current_error
                    max_error_ind = k

            it_error.append(max_error)
            it_ind.append(max_error_ind)

            if Flag == 1:
                return FEAST_output_struct(lambda_m, X_m, i, complexCounter, dimensionReduceCounter, it_error, it_ind, True)

        Y = B @ X_n
	

    return FEAST_output_struct(lambda_m, X_m, maxIter, complexCounter, dimensionReduceCounter, it_error, it_ind, False)
    #maxITer=50 alabilirsin
    #

# Example usage:
# result = FEAST(A, B, M0, Y, Ne, lambdamin, lambdamax, tol, maxIter, options, aei)

