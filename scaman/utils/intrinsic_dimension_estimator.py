import numpy as np

class Estimator:
    """
    Estimator class for estimating the dimension of the subspace spanned by the top eigenvectors of the covariance matrix of the data matrix X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix.
    p : int
        The order of the Jackson kernel.
    nk : int
        The number of iterations of the CGLS algorithm.
    epsilon : float
        The privacy parameter.
    delta : float
        The privacy parameter.
    tv : float
        The target variance.
    
    """
    def __init__(self, X, p, nk, epsilon, delta, tv):
        self.X = X
        self.p = p
        self.nk = nk
        self.epsilon = epsilon
        self.delta = delta
        self.tv = tv
        self.estDim = None
        self.estVar = None
        self.estDimF = None
    
    def randomVectorCount(self, epsilon, delta):
        count = round((2 * (2 + ((8 * np.sqrt(2)) / 3) * epsilon) * np.log(2 / delta)) / (epsilon ** 2))

        return count

    def covarienceTraceEstimator(self, X, epsilon, delta):
        iterCount = self.randomVectorCount(epsilon, delta)

        N, D = X.shape
        estTrace = 0

        for k in range(iterCount):
            z = np.random.choice([-1, 1], size=(D,))
            h = X @ z
            estTrace += h.T @ h

        estTrace = estTrace / (iterCount * (N - 1))

        return estTrace

    def CGLS(self, X, iterCount):
        if iterCount is None:
            iterCount = self.randomVectorCount(self.epsilon, self.delta)

        alphas = np.zeros(iterCount)
        betas = np.zeros(iterCount)
        D = X.shape[1]
        b = X @ np.ones((D, 1))
        b = b / np.linalg.norm(b)

        r = b
        s = X.T @ r
        p = s
        gammao = np.linalg.norm(s) ** 2

        for i in range(iterCount):
            q = X @ p
            alpha = gammao / np.linalg.norm(q) ** 2
            r = r - alpha * q
            s = X.T @ r
            gaman = np.linalg.norm(s) ** 2
            beta = gaman / gammao
            gammao = gaman
            p = s + beta * p

            alphas[i] = alpha
            betas[i] = beta

        return alphas, betas
    
    def Jackson(self, j, p):
        alpha = np.pi / (p + 2)
        g = ((1 - (j / (p + 2))) * np.sin(alpha) * np.cos(j * alpha) + (1 / (p + 2)) * np.cos(alpha) * np.sin(j * alpha)) / np.sin(alpha)
        return g

    def eigenvalueCountEstimator(self,X, p, epsilon, delta, a, b, lmin, lmax):
        iterCount = self.randomVectorCount(epsilon, delta)

        alpha = lmax + lmin
        beta = lmax - lmin

        a = (2 * a - alpha) / beta
        b = (2 * b - alpha) / beta

        #print("a: ", a)
        #print("b: ", b)

        N, D = X.shape
        estCount = 0
        if np.isnan(a) or np.isnan(b):
            raise ValueError("Invalid value of a or b")

        for k in range(iterCount):
            z = np.random.choice([-1, 1], size=D)

            # for j = 0
            g = self.Jackson(0, p)
            gamma = (1 / np.pi) * (np.arccos(a) - np.arccos(b))
            w0 = z
            estCount = estCount + np.dot(z.T, np.dot(g*gamma, w0))

            # for j = 1
            g = self.Jackson(1, p)
            gamma = (2 / np.pi) * (np.sin(np.arccos(a)) - np.sin(np.arccos(b)))
            #w1 = (2*(np.dot(X.T, np.dot(X, z))-(alpha*(N-1)*z))/(beta*(N-1))) 
            w1 = (2*(np.dot(X.T, np.dot(X, z)))-(alpha*(N-1)*z))/(beta*(N-1))
            estCount = estCount + np.dot(z.T, np.dot(g*gamma, w1))

            for j in range(2, p + 1):
                g = self.Jackson(j, p)
                gamma = (2 / np.pi) * ((np.sin(j * np.arccos(a)) - np.sin(j * np.arccos(b))) / j)
                #wnew = 2*(((2*np.dot(X.T, np.dot(X, w1))-(alpha*(N-1)*w1))/(beta*(N-1))))-w0
                wnew = 2*(2*(np.dot(X.T, np.dot(X, w1)))-(alpha*(N-1)*w1))/(beta*(N-1)) - w0
                estCount = estCount + np.dot(z.T, np.dot(g*gamma, wnew))
                w0 = w1
                w1 = wnew
        
        if np.isnan(estCount):
            raise ValueError("Invalid value of estCount")

        estCount = round(estCount / iterCount)

        return estCount

    def estimator(self):
        c = 1.5
        t = self.covarienceTraceEstimator(self.X, self.epsilon, self.delta)
        alphas, betas = self.CGLS(self.X, self.nk)
        T = np.zeros((self.nk, self.nk))
        T[0, 0] = 1 / alphas[0]
        for i in range(1, self.nk):
            T[i, i] = 1 / alphas[i] + (betas[i - 1] / alphas[i - 1])
            T[i, i - 1] = np.sqrt(betas[i - 1]) / alphas[i - 1]
            T[i - 1, i] = np.sqrt(betas[i - 1]) / alphas[i - 1]
        #ritzs = np.flip(np.linalg.eig(T)[0]) / (self.X.shape[0] - 1)
        #print(ritzs)
        # np.linalg.eig does not return sorted results
        ritzs = np.flip(np.sort(np.linalg.eig(T)[0])) / (self.X.shape[0] - 1)
        #print(ritzs) RITZ DEĞERLERINI DONDUR
        alpha = 0
        self.estDim = 0
        estCount = self.eigenvalueCountEstimator(self.X, self.p, self.epsilon, self.delta, ritzs[0], (c - 0.1) * ritzs[0], 0, c * ritzs[0])
        alpha = alpha + ritzs[0] * estCount
        self.estVar = alpha / t
        self.estDim = self.estDim + estCount
        for i in range(self.nk):
            if i == self.nk - 1:
                self.estDimF = self.estDim
                self.estDim = self.X.shape[1]
                return self.estDim, self.estVar, self.estDimF
            estVarOld = self.estVar
            estDimOld = self.estDim
            estCount = self.eigenvalueCountEstimator(self.X, self.p, self.epsilon, self.delta, ritzs[i + 1], ritzs[i], 0, c * ritzs[0])
            if estCount != 1:
                alpha = alpha + ((ritzs[i] + ritzs[i + 1]) / 2) * estCount
            else:
                alpha = alpha + ritzs[i + 1]
            self.estDim = self.estDim + estCount
            self.estVar = alpha / t
            if self.estVar - 0.02 < self.tv and self.estVar + 0.02 > self.tv:
                self.estDimF = self.estDim
                return self.estDim, self.estVar, self.estDimF
            if self.estVar > self.tv:
                break
        self.estDimF = self.estDim
        if i == self.nk - 1:
            return self.estDim, self.estVar, self.estDimF
        if (self.estVar - 0.02) < self.tv:
            return self.estDim, self.estVar, self.estDimF
        if self.estVar == estVarOld:
            return self.estDim, self.estVar, self.estDimF
        self.estDim = round(self.estDim - (self.estVar - self.tv) * ((self.estDim - estDimOld) / (self.estVar - estVarOld)))
        return self.estDim, self.estVar, self.estDimF
        #return ritzs, i değeri: 7 kere döndüyse mesela 7. değer benim için sınır üst değer
        # estDim çalıştır. En küçük 17 değeri bul numpy ile. En küçük 17-18 incele, ortalamasını al ve sınır olarak feast'e ver.