import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc
import time
from scipy.sparse.linalg import linear_operator_2_petsc_shell
import scipy.sparse as sp

def convert_mat_to_petsc(mat, comm=None):
    
    mpi_comm = PETSc.COMM_WORLD if comm is None else comm

    # retrieve before mat is possibly built into sliced matrix
    def convert_mat_to_petsc(mat, comm=None):
        shape = mat.shape

        # create the PETSc matrix with the correct size
        pmat = PETSc.Mat().createAIJ(size=shape, comm=mpi_comm)

        if isinstance(mat, sp.linalg.LinearOperator):
            return linear_operator_2_petsc_shell(mat, comm=comm)

        # populate the PETSc matrix with the values from mat
        for i in range(shape[0]):
            for j in range(shape[1]):
                pmat[i, j] = mat[i, j]

        pmat.setFromOptions()
        pmat.setUp()

    # Assemble the matrix
    pmat.assemblyBegin()
    pmat.assemblyEnd()

    # return the PETSc matrix
    #PETSc.Sys.Print("pmat:")
    #pmat.view()
    return pmat
 

def SlepcEigensolver(data, solver,nev,*args, **kwargs):
    # Convert numpy array to slepc/petsc matrix
    #Eigenvalues and Eigenvectors of Laplacian Matrix
    start_time_petsc_matrix = time.time()
    
    petsc_matH = convert_mat_to_petsc(data, comm =PETSc.COMM_WORLD)
    print("Petsc matrix formed")
    print("Time taken to create petsc matrix: ", time.time() - start_time_petsc_matrix)
    Print = PETSc.Sys.Print
    elapsed_time_petsc_matrix = time.time() - start_time_petsc_matrix

    ########### Calling the Slepc eigen solver #######
    start_time_eig = time.time()
    E = SLEPc.EPS(); E.create(PETSc.COMM_WORLD)

    E.setOperators(petsc_matH)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    '''HEP: Hermitian eigenproblem.
        NHEP: Non-Hermitian eigenproblem.
        GHEP: Generalized Hermitian eigenproblem.
        GNHEP: Generalized Non-Hermitian eigenproblem.
        PGNHEP: Generalized Non-Hermitian eigenproblem with positive definite B.
        GHIEP: Generalized Hermitian-indefinite eigenproblem.'''
    
    eigen_solver_type_dict={
        "KRYLOVSCHUR": SLEPc.EPS.Type.KRYLOVSCHUR,
        "LANCZOS": SLEPc.EPS.Type.LANCZOS,
        "POWER": SLEPc.EPS.Type.POWER,
        "ARNOLDI": SLEPc.EPS.Type.ARNOLDI,
        "GD": SLEPc.EPS.Type.GD,
        "JD": SLEPc.EPS.Type.JD,
        "RQCG": SLEPc.EPS.Type.RQCG,
        "LOBPCG": SLEPc.EPS.Type.LOBPCG
    }
    
    E.setType(eigen_solver_type_dict[solver])
    '''POWER: Power Iteration, Inverse Iteration, RQI.
        SUBSPACE: Subspace Iteration. (+)
        ARNOLDI: Arnoldi. (+)
        LANCZOS: Lanczos. (!) (EPS_HEP, EPS_GHEP)(Krylovschur method can also implement lanczos if symm. and herm.)
        KRYLOVSCHUR: Krylov-Schur (default). (+)
        GD: Generalized Davidson. (+)
        JD: Jacobi-Davidson. (+)(working slowly)
        RQCG: Rayleigh Quotient Conjugate Gradient. (+) (increase the iteration number, any eig couldn't found.)
        LOBPCG: Locally Optimal Block Preconditioned Conjugate Gradient. (install external drivers)
        CISS: Contour Integral Spectrum Slicing. (works for all eigenvalues, not fol partial dec.)
        LYAPII: Lyapunov inverse iteration.(works for all eigenvalues, not fol partial dec.)
        LAPACK: Wrappers to dense eigensolvers in Lapack. (+)(works and finds all eigenvalues)'''
        
    # Set the solver type to Lanczos
    if solver == 'LANCZOS':
        E.setType(SLEPc.EPS.Type.LANCZOS)

    E.setDimensions(nev)
    tol = kwargs.get('tol', 1e-3)  # Use the provided tol value if available, otherwise default to 1e-3
    E.setTolerances(tol=tol)
    
    magnitude = kwargs.get('magnitude', 'largest') # Use the provided magnitude value if available, otherwise default to largest
    if magnitude == "largest":
        E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    elif magnitude == "smallest":    
        E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    
    

    E.solve()

    end_time_eig = time.time() - start_time_eig

    Print()
    Print("******************************")
    Print("*** SLEPc Solution Results ***")
    Print("******************************")
    Print()

    its = E.getIterationNumber()
    Print("Number of iterations of the method: %d" % its)

    eps_type = E.getType()
    Print("Solution method: %s" % eps_type)

    nev, ncv, mpd = E.getDimensions()
    Print("Number of requested eigenvalues: %d" % nev)

    tol, maxit = E.getTolerances()
    Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

    nconv = E.getConverged()
    Print("Number of converged eigenpairs %d" % nconv)

    Print("End time for eigen solver: ", end_time_eig)


    global eigenvalues,eigenvectors
    if nconv > 0:
        # Create the results vectors
        vr, wr = petsc_matH.getVecs()
        vi, wi = petsc_matH.getVecs()
        # Store eigenvalues and eigenvectors as an array

        eigenvalues=[]
        eigenvectors=[]

        #Print()
        #Print("Eigenpairs:")
        #Print("        k          ||Ax-kx||/||kx|| ")
        #Print("----------------- ------------------")
        for i in range(nconv):
            k = E.getEigenpair(i, vr, vi)
            error = E.computeError(i)

            eigenvalues.append(k)
            eigenvectors.append([complex(vr0, vi0) for vr0, vi0 in zip(vr.getArray(), vi.getArray())])

            '''if k.imag != 0.0:
                Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
            else:
                Print(" %12f      %12g" % (k.real, error))'''

        #Print()
        eigenvalues = np.asarray(eigenvalues)
        eigenvectors = np.asarray(eigenvectors)
        return eigenvalues,eigenvectors
    else:
        Print('SLEPc could not find any eigenpairs with given n_dim %d and solver %s'  %(nev,solver))

