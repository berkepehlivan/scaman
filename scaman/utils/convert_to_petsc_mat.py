def convert_mat_to_petsc(mat, comm=None):
    mpi_comm = PETSc.COMM_WORLD if comm is None else comm

    # retrieve before mat is possibly built into sliced matrix
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
    PETSc.Sys.Print("pmat:")
    pmat.view()
    return pmat