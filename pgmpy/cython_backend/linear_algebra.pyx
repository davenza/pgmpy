#cython: boundscheck=False, wraparound=False, cdivision=True
from libc.stdlib cimport calloc, malloc, free
from libc.string cimport memcpy

cimport scipy.linalg.cython_lapack as cython_lapack
cimport scipy.linalg.cython_blas as cython_blas

import numpy as np

cdef inverse_dgetr(double[:,:] mat, double[:,:] out_inv):
    """
    https://stackoverflow.com/questions/3519959/computing-the-inverse-of-a-matrix-using-lapack-in-c
    :param mat: 
    :return: 
    """
    # you should probably check that mat.shape[0]==mat.shape[1]
    # and that mat is actually a float64 array
    cdef int k = mat.shape[0]

    if (&mat[0,0] != &out_inv[0,0]):
        memcpy(&out_inv[0,0], &mat[0,0], k*k*sizeof(double))

    cdef int info = 0

    cdef double* mat_pointer = &out_inv[0,0]

    cdef int* piv_pointer = <int*>malloc(sizeof(int)*k)

    workspace = <double*>malloc(sizeof(double)*k*k)

    cdef int len_workspace = k*k
                # this is uninitialised (the contents are arbitrary)
                # but that's OK because it's used as an output
    try:
        cython_lapack.dgetrf(&k,&k,mat_pointer,&k,piv_pointer,&info)
        cython_lapack.dgetri(&k,mat_pointer,&k,piv_pointer, workspace, &len_workspace, &info)
        # you should check info to ensure it's worked
    finally:
        free(piv_pointer) # the "try ... finally" ensures that this is freed
        free(workspace) # the "try ... finally" ensures that this is freed


cdef inverse_dpotr(double[:,:] mat, double[:,:] out_inv):
    """
    This function inverses symmetric positive definite matrices.
    :param mat: 
    :return: 
    """
    # you should probably check that mat.shape[0]==mat.shape[1]
    # and that mat is actually a float64 array
    cdef int k = mat.shape[0]
    cdef char uplo = 'U'

    if (&mat[0,0] != &out_inv[0,0]):
        memcpy(&out_inv[0,0], &mat[0,0], k*k*sizeof(double))

    cdef int info = 0

    cdef double* mat_pointer = &out_inv[0,0]
    # I suspect you should be doing a check that mat_pointer has been assigned

    cython_lapack.dpotrf(&uplo,&k,mat_pointer,&k,&info)
    cython_lapack.dpotri(&uplo,&k,mat_pointer,&k,&info)


# ######################################################################################################
# ######################################################################################################
#
# The code of the following functions can be generated with the following python functions.
#
# det_str generates a string of multiplications to compute the determinant of the matrix with rows and columns indices.
# adjugate_str generates a string of multiplications that generates the adjugate matrix of order n.
#
# For example, to obtain the adjugate matrix of order 5, you should call the functions as:
#
# rows = cols = list(range(5))
# print(adjugate_str(rows, cols, "mat", "out_inv"))
#
#
# ######################################################################################################
# ######################################################################################################

# def det_str(rows, cols, positive, in_str):
#     if len(rows) == 2 and len(cols) == 2:
#         if positive:
#             return in_str + "[" + str(rows[0]) + "," + str(cols[0]) + "]*" + \
#                     in_str + "[" + str(rows[1]) + "," + str(cols[1]) + "] - " + \
#                        in_str + "[" + str(rows[0]) + "," + str(cols[1]) + "]*" + \
#                        in_str + "[" + str(rows[1]) + "," + str(cols[0]) + "]"
#         else:
#             return in_str + "[" + str(rows[0]) + "," + str(cols[1]) + "]*" + \
#                     in_str + "[" + str(rows[1]) + "," + str(cols[0]) + "] - " + \
#                     in_str + "[" + str(rows[0]) + "," + str(cols[0]) + "]*" + \
#                     in_str + "[" + str(rows[1]) + "," + str(cols[1]) + "]"
#
#     else:
#         output_str = ""
#         for i in range(len(cols)):
#             removed_rows = rows.copy()
#             del removed_rows[0]
#             removed_cols = cols.copy()
#             del removed_cols[i]
#             if (positive and i % 2 == 0) or ((not positive) and i % 2 == 1):
#                 output_str += (in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(" + det_str(removed_rows, removed_cols, True, in_str) + ")")
#             else:
#                 output_str += (in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(" + det_str(removed_rows, removed_cols, False, in_str) + ")")
#             if i < len(cols)-1:
#                 output_str += " + \n\t\t"
#
#         return output_str
#
# def adjugate_str(rows, cols, in_str, out_str):
#     res = ""
#     for i in range(len(rows)):
#         for j in range(len(cols)):
#             removed_rows = rows.copy()
#             del removed_rows[i]
#             removed_cols = cols.copy()
#             del removed_cols[j]
#             if ((i+j) % 2) == 0:
#                 res += out_str + "[" + str(i) + "," + str(j) + "] = (" + det_str(removed_cols, removed_rows, True, in_str) + ")\n"
#             else:
#                 res +=  out_str + "[" + str(i) + "," + str(j) + "] = -(" + det_str(removed_cols, removed_rows, True, in_str)  + ")\n"
#
#             res += '\n'
#
#     return res


cdef void inverse(double[:,:] mat, double[:,:] out_inv):
    """
    Computes the inverse of a matrix `mat` in a matrix `out_inv`.
    
    **IMPORTANT NOTE: mat should be different to out_inv**.
    """
    cdef Py_ssize_t k = mat.shape[0], i, jinverse_dpotrf
    cdef double inv_det
    # TODO: Check out_inv is different pointer to mat.
    if k == 2:
        inv_det = 1.0 / (mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0])
        out_inv[0,0] = mat[1,1] * inv_det
        out_inv[0,1] = -mat[0,1] * inv_det
        out_inv[1,0] = -mat[1,0] * inv_det
        out_inv[1,1] = mat[0,0] * inv_det
    elif k == 3:
        inv_det = 1.0 / (mat[0,0]*(mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) +
                         mat[0,1]*(mat[1,2]*mat[2,0] - mat[1,0]*mat[2,2]) +
                         mat[0,2]*(mat[1,0]*mat[2,1] - mat[1,1]*mat[2,0]) )

        out_inv[0,0] = (mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) * inv_det
        out_inv[0,1] = -(mat[0,1]*mat[2,2] - mat[0,2]*mat[2,1]) * inv_det
        out_inv[0,2] = (mat[0,1]*mat[1,2] - mat[0,2]*mat[1,1]) * inv_det
        out_inv[1,0] = -(mat[1,0]*mat[2,2] - mat[1,2]*mat[2,0]) * inv_det
        out_inv[1,1] = (mat[0,0]*mat[2,2] - mat[0,2]*mat[2,0]) * inv_det
        out_inv[1,2] = -(mat[0,0]*mat[1,2] - mat[0,2]*mat[1,0]) * inv_det
        out_inv[2,0] = (mat[1,0]*mat[2,1] - mat[1,1]*mat[2,0]) * inv_det
        out_inv[2,1] = -(mat[0,0]*mat[2,1] - mat[0,1]*mat[2,0]) * inv_det
        out_inv[2,2] = (mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0]) * inv_det
    elif k == 4:
        out_inv[0,0] = (mat[1,1]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[1,2]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3]) +
                        mat[1,3]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]) )

        out_inv[0,1] = -(mat[0,1]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[0,2]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3]) +
                        mat[0,3]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]) )

        out_inv[0,2] = (mat[0,1]*(mat[1,2]*mat[3,3] - mat[1,3]*mat[3,2]) +
                        mat[0,2]*(mat[1,3]*mat[3,1] - mat[1,1]*mat[3,3]) +
                        mat[0,3]*(mat[1,1]*mat[3,2] - mat[1,2]*mat[3,1]) )

        out_inv[0,3] = -(mat[0,1]*(mat[1,2]*mat[2,3] - mat[1,3]*mat[2,2]) +
                        mat[0,2]*(mat[1,3]*mat[2,1] - mat[1,1]*mat[2,3]) +
                        mat[0,3]*(mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) )

        out_inv[1,0] = -(mat[1,0]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[1,2]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3]) +
                        mat[1,3]*(mat[2,0]*mat[3,2] - mat[2,2]*mat[3,0]) )

        out_inv[1,1] = (mat[0,0]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[0,2]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3]) +
                        mat[0,3]*(mat[2,0]*mat[3,2] - mat[2,2]*mat[3,0]) )

        out_inv[1,2] = -(mat[0,0]*(mat[1,2]*mat[3,3] - mat[1,3]*mat[3,2]) +
                        mat[0,2]*(mat[1,3]*mat[3,0] - mat[1,0]*mat[3,3]) +
                        mat[0,3]*(mat[1,0]*mat[3,2] - mat[1,2]*mat[3,0]) )

        out_inv[1,3] = (mat[0,0]*(mat[1,2]*mat[2,3] - mat[1,3]*mat[2,2]) +
                        mat[0,2]*(mat[1,3]*mat[2,0] - mat[1,0]*mat[2,3]) +
                        mat[0,3]*(mat[1,0]*mat[2,2] - mat[1,2]*mat[2,0]) )

        out_inv[2,0] = (mat[1,0]*(mat[2,1]*mat[3,3] - mat[2,3]*mat[3,1]) +
                        mat[1,1]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3]) +
                        mat[1,3]*(mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0]) )

        out_inv[2,1] = -(mat[0,0]*(mat[2,1]*mat[3,3] - mat[2,3]*mat[3,1]) +
                        mat[0,1]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3]) +
                        mat[0,3]*(mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0]) )

        out_inv[2,2] = (mat[0,0]*(mat[1,1]*mat[3,3] - mat[1,3]*mat[3,1]) +
                        mat[0,1]*(mat[1,3]*mat[3,0] - mat[1,0]*mat[3,3]) +
                        mat[0,3]*(mat[1,0]*mat[3,1] - mat[1,1]*mat[3,0]) )

        out_inv[2,3] = -(mat[0,0]*(mat[1,1]*mat[2,3] - mat[1,3]*mat[2,1]) +
                         mat[0,1]*(mat[1,3]*mat[2,0] - mat[1,0]*mat[2,3]) +
                         mat[0,3]*(mat[1,0]*mat[2,1] - mat[1,1]*mat[2,0]) )

        out_inv[3,0] = -(mat[1,0]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]) +
                         mat[1,1]*(mat[2,2]*mat[3,0] - mat[2,0]*mat[3,2]) +
                         mat[1,2]*(mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0]) )

        out_inv[3,1] = (mat[0,0]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]) +
                        mat[0,1]*(mat[2,2]*mat[3,0] - mat[2,0]*mat[3,2]) +
                        mat[0,2]*(mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0]) )

        out_inv[3,2] = -(mat[0,0]*(mat[1,1]*mat[3,2] - mat[1,2]*mat[3,1]) +
                         mat[0,1]*(mat[1,2]*mat[3,0] - mat[1,0]*mat[3,2]) +
                         mat[0,2]*(mat[1,0]*mat[3,1] - mat[1,1]*mat[3,0]) )

        out_inv[3,3] = (mat[0,0]*(mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) +
                        mat[0,1]*(mat[1,2]*mat[2,0] - mat[1,0]*mat[2,2]) +
                        mat[0,2]*(mat[1,0]*mat[2,1] - mat[1,1]*mat[2,0]) )

        inv_det = 1.0/(mat[0,0]*out_inv[0,0] + mat[1,0]*out_inv[0,1] + mat[2,0]*out_inv[0,2] + mat[3,0]*out_inv[0,3])

        for i in range(k):
            for j in range(k):
                out_inv[i,j] *= inv_det

    elif k==5:
        out_inv[0,0] = (mat[1,1]*(mat[2,2]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[2,3]*(mat[3,4]*mat[4,2] - mat[3,2]*mat[4,4]) +
                mat[2,4]*(mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2])) +
                mat[1,2]*(mat[2,1]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[2,3]*(mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]) +
                mat[2,4]*(mat[3,3]*mat[4,1] - mat[3,1]*mat[4,3])) +
                mat[1,3]*(mat[2,1]*(mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]) +
                mat[2,2]*(mat[3,4]*mat[4,1] - mat[3,1]*mat[4,4]) +
                mat[2,4]*(mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1])) +
                mat[1,4]*(mat[2,1]*(mat[3,3]*mat[4,2] - mat[3,2]*mat[4,3]) +
                mat[2,2]*(mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1]) +
                mat[2,3]*(mat[3,2]*mat[4,1] - mat[3,1]*mat[4,2])))

        out_inv[0,1] = -(mat[0,1]*(mat[2,2]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[2,3]*(mat[3,4]*mat[4,2] - mat[3,2]*mat[4,4]) +
                mat[2,4]*(mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2])) +
                mat[0,2]*(mat[2,1]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[2,3]*(mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]) +
                mat[2,4]*(mat[3,3]*mat[4,1] - mat[3,1]*mat[4,3])) +
                mat[0,3]*(mat[2,1]*(mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]) +
                mat[2,2]*(mat[3,4]*mat[4,1] - mat[3,1]*mat[4,4]) +
                mat[2,4]*(mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1])) +
                mat[0,4]*(mat[2,1]*(mat[3,3]*mat[4,2] - mat[3,2]*mat[4,3]) +
                mat[2,2]*(mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1]) +
                mat[2,3]*(mat[3,2]*mat[4,1] - mat[3,1]*mat[4,2])))

        out_inv[0,2] = (mat[0,1]*(mat[1,2]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[1,3]*(mat[3,4]*mat[4,2] - mat[3,2]*mat[4,4]) +
                mat[1,4]*(mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2])) +
                mat[0,2]*(mat[1,1]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[1,3]*(mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]) +
                mat[1,4]*(mat[3,3]*mat[4,1] - mat[3,1]*mat[4,3])) +
                mat[0,3]*(mat[1,1]*(mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]) +
                mat[1,2]*(mat[3,4]*mat[4,1] - mat[3,1]*mat[4,4]) +
                mat[1,4]*(mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1])) +
                mat[0,4]*(mat[1,1]*(mat[3,3]*mat[4,2] - mat[3,2]*mat[4,3]) +
                mat[1,2]*(mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1]) +
                mat[1,3]*(mat[3,2]*mat[4,1] - mat[3,1]*mat[4,2])))

        out_inv[0,3] = -(mat[0,1]*(mat[1,2]*(mat[2,3]*mat[4,4] - mat[2,4]*mat[4,3]) +
                mat[1,3]*(mat[2,4]*mat[4,2] - mat[2,2]*mat[4,4]) +
                mat[1,4]*(mat[2,2]*mat[4,3] - mat[2,3]*mat[4,2])) +
                mat[0,2]*(mat[1,1]*(mat[2,4]*mat[4,3] - mat[2,3]*mat[4,4]) +
                mat[1,3]*(mat[2,1]*mat[4,4] - mat[2,4]*mat[4,1]) +
                mat[1,4]*(mat[2,3]*mat[4,1] - mat[2,1]*mat[4,3])) +
                mat[0,3]*(mat[1,1]*(mat[2,2]*mat[4,4] - mat[2,4]*mat[4,2]) +
                mat[1,2]*(mat[2,4]*mat[4,1] - mat[2,1]*mat[4,4]) +
                mat[1,4]*(mat[2,1]*mat[4,2] - mat[2,2]*mat[4,1])) +
                mat[0,4]*(mat[1,1]*(mat[2,3]*mat[4,2] - mat[2,2]*mat[4,3]) +
                mat[1,2]*(mat[2,1]*mat[4,3] - mat[2,3]*mat[4,1]) +
                mat[1,3]*(mat[2,2]*mat[4,1] - mat[2,1]*mat[4,2])))

        out_inv[0,4] = (mat[0,1]*(mat[1,2]*(mat[2,3]*mat[3,4] - mat[2,4]*mat[3,3]) +
                mat[1,3]*(mat[2,4]*mat[3,2] - mat[2,2]*mat[3,4]) +
                mat[1,4]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2])) +
                mat[0,2]*(mat[1,1]*(mat[2,4]*mat[3,3] - mat[2,3]*mat[3,4]) +
                mat[1,3]*(mat[2,1]*mat[3,4] - mat[2,4]*mat[3,1]) +
                mat[1,4]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3])) +
                mat[0,3]*(mat[1,1]*(mat[2,2]*mat[3,4] - mat[2,4]*mat[3,2]) +
                mat[1,2]*(mat[2,4]*mat[3,1] - mat[2,1]*mat[3,4]) +
                mat[1,4]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1])) +
                mat[0,4]*(mat[1,1]*(mat[2,3]*mat[3,2] - mat[2,2]*mat[3,3]) +
                mat[1,2]*(mat[2,1]*mat[3,3] - mat[2,3]*mat[3,1]) +
                mat[1,3]*(mat[2,2]*mat[3,1] - mat[2,1]*mat[3,2])))

        out_inv[1,0] = -(mat[1,0]*(mat[2,2]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[2,3]*(mat[3,4]*mat[4,2] - mat[3,2]*mat[4,4]) +
                mat[2,4]*(mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2])) +
                mat[1,2]*(mat[2,0]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[2,3]*(mat[3,0]*mat[4,4] - mat[3,4]*mat[4,0]) +
                mat[2,4]*(mat[3,3]*mat[4,0] - mat[3,0]*mat[4,3])) +
                mat[1,3]*(mat[2,0]*(mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]) +
                mat[2,2]*(mat[3,4]*mat[4,0] - mat[3,0]*mat[4,4]) +
                mat[2,4]*(mat[3,0]*mat[4,2] - mat[3,2]*mat[4,0])) +
                mat[1,4]*(mat[2,0]*(mat[3,3]*mat[4,2] - mat[3,2]*mat[4,3]) +
                mat[2,2]*(mat[3,0]*mat[4,3] - mat[3,3]*mat[4,0]) +
                mat[2,3]*(mat[3,2]*mat[4,0] - mat[3,0]*mat[4,2])))

        out_inv[1,1] = (mat[0,0]*(mat[2,2]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[2,3]*(mat[3,4]*mat[4,2] - mat[3,2]*mat[4,4]) +
                mat[2,4]*(mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2])) +
                mat[0,2]*(mat[2,0]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[2,3]*(mat[3,0]*mat[4,4] - mat[3,4]*mat[4,0]) +
                mat[2,4]*(mat[3,3]*mat[4,0] - mat[3,0]*mat[4,3])) +
                mat[0,3]*(mat[2,0]*(mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]) +
                mat[2,2]*(mat[3,4]*mat[4,0] - mat[3,0]*mat[4,4]) +
                mat[2,4]*(mat[3,0]*mat[4,2] - mat[3,2]*mat[4,0])) +
                mat[0,4]*(mat[2,0]*(mat[3,3]*mat[4,2] - mat[3,2]*mat[4,3]) +
                mat[2,2]*(mat[3,0]*mat[4,3] - mat[3,3]*mat[4,0]) +
                mat[2,3]*(mat[3,2]*mat[4,0] - mat[3,0]*mat[4,2])))

        out_inv[1,2] = -(mat[0,0]*(mat[1,2]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[1,3]*(mat[3,4]*mat[4,2] - mat[3,2]*mat[4,4]) +
                mat[1,4]*(mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2])) +
                mat[0,2]*(mat[1,0]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[1,3]*(mat[3,0]*mat[4,4] - mat[3,4]*mat[4,0]) +
                mat[1,4]*(mat[3,3]*mat[4,0] - mat[3,0]*mat[4,3])) +
                mat[0,3]*(mat[1,0]*(mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]) +
                mat[1,2]*(mat[3,4]*mat[4,0] - mat[3,0]*mat[4,4]) +
                mat[1,4]*(mat[3,0]*mat[4,2] - mat[3,2]*mat[4,0])) +
                mat[0,4]*(mat[1,0]*(mat[3,3]*mat[4,2] - mat[3,2]*mat[4,3]) +
                mat[1,2]*(mat[3,0]*mat[4,3] - mat[3,3]*mat[4,0]) +
                mat[1,3]*(mat[3,2]*mat[4,0] - mat[3,0]*mat[4,2])))

        out_inv[1,3] = (mat[0,0]*(mat[1,2]*(mat[2,3]*mat[4,4] - mat[2,4]*mat[4,3]) +
                mat[1,3]*(mat[2,4]*mat[4,2] - mat[2,2]*mat[4,4]) +
                mat[1,4]*(mat[2,2]*mat[4,3] - mat[2,3]*mat[4,2])) +
                mat[0,2]*(mat[1,0]*(mat[2,4]*mat[4,3] - mat[2,3]*mat[4,4]) +
                mat[1,3]*(mat[2,0]*mat[4,4] - mat[2,4]*mat[4,0]) +
                mat[1,4]*(mat[2,3]*mat[4,0] - mat[2,0]*mat[4,3])) +
                mat[0,3]*(mat[1,0]*(mat[2,2]*mat[4,4] - mat[2,4]*mat[4,2]) +
                mat[1,2]*(mat[2,4]*mat[4,0] - mat[2,0]*mat[4,4]) +
                mat[1,4]*(mat[2,0]*mat[4,2] - mat[2,2]*mat[4,0])) +
                mat[0,4]*(mat[1,0]*(mat[2,3]*mat[4,2] - mat[2,2]*mat[4,3]) +
                mat[1,2]*(mat[2,0]*mat[4,3] - mat[2,3]*mat[4,0]) +
                mat[1,3]*(mat[2,2]*mat[4,0] - mat[2,0]*mat[4,2])))

        out_inv[1,4] = -(mat[0,0]*(mat[1,2]*(mat[2,3]*mat[3,4] - mat[2,4]*mat[3,3]) +
                mat[1,3]*(mat[2,4]*mat[3,2] - mat[2,2]*mat[3,4]) +
                mat[1,4]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2])) +
                mat[0,2]*(mat[1,0]*(mat[2,4]*mat[3,3] - mat[2,3]*mat[3,4]) +
                mat[1,3]*(mat[2,0]*mat[3,4] - mat[2,4]*mat[3,0]) +
                mat[1,4]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3])) +
                mat[0,3]*(mat[1,0]*(mat[2,2]*mat[3,4] - mat[2,4]*mat[3,2]) +
                mat[1,2]*(mat[2,4]*mat[3,0] - mat[2,0]*mat[3,4]) +
                mat[1,4]*(mat[2,0]*mat[3,2] - mat[2,2]*mat[3,0])) +
                mat[0,4]*(mat[1,0]*(mat[2,3]*mat[3,2] - mat[2,2]*mat[3,3]) +
                mat[1,2]*(mat[2,0]*mat[3,3] - mat[2,3]*mat[3,0]) +
                mat[1,3]*(mat[2,2]*mat[3,0] - mat[2,0]*mat[3,2])))

        out_inv[2,0] = (mat[1,0]*(mat[2,1]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[2,3]*(mat[3,4]*mat[4,1] - mat[3,1]*mat[4,4]) +
                mat[2,4]*(mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1])) +
                mat[1,1]*(mat[2,0]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[2,3]*(mat[3,0]*mat[4,4] - mat[3,4]*mat[4,0]) +
                mat[2,4]*(mat[3,3]*mat[4,0] - mat[3,0]*mat[4,3])) +
                mat[1,3]*(mat[2,0]*(mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]) +
                mat[2,1]*(mat[3,4]*mat[4,0] - mat[3,0]*mat[4,4]) +
                mat[2,4]*(mat[3,0]*mat[4,1] - mat[3,1]*mat[4,0])) +
                mat[1,4]*(mat[2,0]*(mat[3,3]*mat[4,1] - mat[3,1]*mat[4,3]) +
                mat[2,1]*(mat[3,0]*mat[4,3] - mat[3,3]*mat[4,0]) +
                mat[2,3]*(mat[3,1]*mat[4,0] - mat[3,0]*mat[4,1])))

        out_inv[2,1] = -(mat[0,0]*(mat[2,1]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[2,3]*(mat[3,4]*mat[4,1] - mat[3,1]*mat[4,4]) +
                mat[2,4]*(mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1])) +
                mat[0,1]*(mat[2,0]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[2,3]*(mat[3,0]*mat[4,4] - mat[3,4]*mat[4,0]) +
                mat[2,4]*(mat[3,3]*mat[4,0] - mat[3,0]*mat[4,3])) +
                mat[0,3]*(mat[2,0]*(mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]) +
                mat[2,1]*(mat[3,4]*mat[4,0] - mat[3,0]*mat[4,4]) +
                mat[2,4]*(mat[3,0]*mat[4,1] - mat[3,1]*mat[4,0])) +
                mat[0,4]*(mat[2,0]*(mat[3,3]*mat[4,1] - mat[3,1]*mat[4,3]) +
                mat[2,1]*(mat[3,0]*mat[4,3] - mat[3,3]*mat[4,0]) +
                mat[2,3]*(mat[3,1]*mat[4,0] - mat[3,0]*mat[4,1])))

        out_inv[2,2] = (mat[0,0]*(mat[1,1]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[1,3]*(mat[3,4]*mat[4,1] - mat[3,1]*mat[4,4]) +
                mat[1,4]*(mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1])) +
                mat[0,1]*(mat[1,0]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[1,3]*(mat[3,0]*mat[4,4] - mat[3,4]*mat[4,0]) +
                mat[1,4]*(mat[3,3]*mat[4,0] - mat[3,0]*mat[4,3])) +
                mat[0,3]*(mat[1,0]*(mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]) +
                mat[1,1]*(mat[3,4]*mat[4,0] - mat[3,0]*mat[4,4]) +
                mat[1,4]*(mat[3,0]*mat[4,1] - mat[3,1]*mat[4,0])) +
                mat[0,4]*(mat[1,0]*(mat[3,3]*mat[4,1] - mat[3,1]*mat[4,3]) +
                mat[1,1]*(mat[3,0]*mat[4,3] - mat[3,3]*mat[4,0]) +
                mat[1,3]*(mat[3,1]*mat[4,0] - mat[3,0]*mat[4,1])))

        out_inv[2,3] = -(mat[0,0]*(mat[1,1]*(mat[2,3]*mat[4,4] - mat[2,4]*mat[4,3]) +
                mat[1,3]*(mat[2,4]*mat[4,1] - mat[2,1]*mat[4,4]) +
                mat[1,4]*(mat[2,1]*mat[4,3] - mat[2,3]*mat[4,1])) +
                mat[0,1]*(mat[1,0]*(mat[2,4]*mat[4,3] - mat[2,3]*mat[4,4]) +
                mat[1,3]*(mat[2,0]*mat[4,4] - mat[2,4]*mat[4,0]) +
                mat[1,4]*(mat[2,3]*mat[4,0] - mat[2,0]*mat[4,3])) +
                mat[0,3]*(mat[1,0]*(mat[2,1]*mat[4,4] - mat[2,4]*mat[4,1]) +
                mat[1,1]*(mat[2,4]*mat[4,0] - mat[2,0]*mat[4,4]) +
                mat[1,4]*(mat[2,0]*mat[4,1] - mat[2,1]*mat[4,0])) +
                mat[0,4]*(mat[1,0]*(mat[2,3]*mat[4,1] - mat[2,1]*mat[4,3]) +
                mat[1,1]*(mat[2,0]*mat[4,3] - mat[2,3]*mat[4,0]) +
                mat[1,3]*(mat[2,1]*mat[4,0] - mat[2,0]*mat[4,1])))

        out_inv[2,4] = (mat[0,0]*(mat[1,1]*(mat[2,3]*mat[3,4] - mat[2,4]*mat[3,3]) +
                mat[1,3]*(mat[2,4]*mat[3,1] - mat[2,1]*mat[3,4]) +
                mat[1,4]*(mat[2,1]*mat[3,3] - mat[2,3]*mat[3,1])) +
                mat[0,1]*(mat[1,0]*(mat[2,4]*mat[3,3] - mat[2,3]*mat[3,4]) +
                mat[1,3]*(mat[2,0]*mat[3,4] - mat[2,4]*mat[3,0]) +
                mat[1,4]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3])) +
                mat[0,3]*(mat[1,0]*(mat[2,1]*mat[3,4] - mat[2,4]*mat[3,1]) +
                mat[1,1]*(mat[2,4]*mat[3,0] - mat[2,0]*mat[3,4]) +
                mat[1,4]*(mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0])) +
                mat[0,4]*(mat[1,0]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3]) +
                mat[1,1]*(mat[2,0]*mat[3,3] - mat[2,3]*mat[3,0]) +
                mat[1,3]*(mat[2,1]*mat[3,0] - mat[2,0]*mat[3,1])))

        out_inv[3,0] = -(mat[1,0]*(mat[2,1]*(mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]) +
                mat[2,2]*(mat[3,4]*mat[4,1] - mat[3,1]*mat[4,4]) +
                mat[2,4]*(mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1])) +
                mat[1,1]*(mat[2,0]*(mat[3,4]*mat[4,2] - mat[3,2]*mat[4,4]) +
                mat[2,2]*(mat[3,0]*mat[4,4] - mat[3,4]*mat[4,0]) +
                mat[2,4]*(mat[3,2]*mat[4,0] - mat[3,0]*mat[4,2])) +
                mat[1,2]*(mat[2,0]*(mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]) +
                mat[2,1]*(mat[3,4]*mat[4,0] - mat[3,0]*mat[4,4]) +
                mat[2,4]*(mat[3,0]*mat[4,1] - mat[3,1]*mat[4,0])) +
                mat[1,4]*(mat[2,0]*(mat[3,2]*mat[4,1] - mat[3,1]*mat[4,2]) +
                mat[2,1]*(mat[3,0]*mat[4,2] - mat[3,2]*mat[4,0]) +
                mat[2,2]*(mat[3,1]*mat[4,0] - mat[3,0]*mat[4,1])))

        out_inv[3,1] = (mat[0,0]*(mat[2,1]*(mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]) +
                mat[2,2]*(mat[3,4]*mat[4,1] - mat[3,1]*mat[4,4]) +
                mat[2,4]*(mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1])) +
                mat[0,1]*(mat[2,0]*(mat[3,4]*mat[4,2] - mat[3,2]*mat[4,4]) +
                mat[2,2]*(mat[3,0]*mat[4,4] - mat[3,4]*mat[4,0]) +
                mat[2,4]*(mat[3,2]*mat[4,0] - mat[3,0]*mat[4,2])) +
                mat[0,2]*(mat[2,0]*(mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]) +
                mat[2,1]*(mat[3,4]*mat[4,0] - mat[3,0]*mat[4,4]) +
                mat[2,4]*(mat[3,0]*mat[4,1] - mat[3,1]*mat[4,0])) +
                mat[0,4]*(mat[2,0]*(mat[3,2]*mat[4,1] - mat[3,1]*mat[4,2]) +
                mat[2,1]*(mat[3,0]*mat[4,2] - mat[3,2]*mat[4,0]) +
                mat[2,2]*(mat[3,1]*mat[4,0] - mat[3,0]*mat[4,1])))

        out_inv[3,2] = -(mat[0,0]*(mat[1,1]*(mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]) +
                mat[1,2]*(mat[3,4]*mat[4,1] - mat[3,1]*mat[4,4]) +
                mat[1,4]*(mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1])) +
                mat[0,1]*(mat[1,0]*(mat[3,4]*mat[4,2] - mat[3,2]*mat[4,4]) +
                mat[1,2]*(mat[3,0]*mat[4,4] - mat[3,4]*mat[4,0]) +
                mat[1,4]*(mat[3,2]*mat[4,0] - mat[3,0]*mat[4,2])) +
                mat[0,2]*(mat[1,0]*(mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]) +
                mat[1,1]*(mat[3,4]*mat[4,0] - mat[3,0]*mat[4,4]) +
                mat[1,4]*(mat[3,0]*mat[4,1] - mat[3,1]*mat[4,0])) +
                mat[0,4]*(mat[1,0]*(mat[3,2]*mat[4,1] - mat[3,1]*mat[4,2]) +
                mat[1,1]*(mat[3,0]*mat[4,2] - mat[3,2]*mat[4,0]) +
                mat[1,2]*(mat[3,1]*mat[4,0] - mat[3,0]*mat[4,1])))

        out_inv[3,3] = (mat[0,0]*(mat[1,1]*(mat[2,2]*mat[4,4] - mat[2,4]*mat[4,2]) +
                mat[1,2]*(mat[2,4]*mat[4,1] - mat[2,1]*mat[4,4]) +
                mat[1,4]*(mat[2,1]*mat[4,2] - mat[2,2]*mat[4,1])) +
                mat[0,1]*(mat[1,0]*(mat[2,4]*mat[4,2] - mat[2,2]*mat[4,4]) +
                mat[1,2]*(mat[2,0]*mat[4,4] - mat[2,4]*mat[4,0]) +
                mat[1,4]*(mat[2,2]*mat[4,0] - mat[2,0]*mat[4,2])) +
                mat[0,2]*(mat[1,0]*(mat[2,1]*mat[4,4] - mat[2,4]*mat[4,1]) +
                mat[1,1]*(mat[2,4]*mat[4,0] - mat[2,0]*mat[4,4]) +
                mat[1,4]*(mat[2,0]*mat[4,1] - mat[2,1]*mat[4,0])) +
                mat[0,4]*(mat[1,0]*(mat[2,2]*mat[4,1] - mat[2,1]*mat[4,2]) +
                mat[1,1]*(mat[2,0]*mat[4,2] - mat[2,2]*mat[4,0]) +
                mat[1,2]*(mat[2,1]*mat[4,0] - mat[2,0]*mat[4,1])))

        out_inv[3,4] = -(mat[0,0]*(mat[1,1]*(mat[2,2]*mat[3,4] - mat[2,4]*mat[3,2]) +
                mat[1,2]*(mat[2,4]*mat[3,1] - mat[2,1]*mat[3,4]) +
                mat[1,4]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1])) +
                mat[0,1]*(mat[1,0]*(mat[2,4]*mat[3,2] - mat[2,2]*mat[3,4]) +
                mat[1,2]*(mat[2,0]*mat[3,4] - mat[2,4]*mat[3,0]) +
                mat[1,4]*(mat[2,2]*mat[3,0] - mat[2,0]*mat[3,2])) +
                mat[0,2]*(mat[1,0]*(mat[2,1]*mat[3,4] - mat[2,4]*mat[3,1]) +
                mat[1,1]*(mat[2,4]*mat[3,0] - mat[2,0]*mat[3,4]) +
                mat[1,4]*(mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0])) +
                mat[0,4]*(mat[1,0]*(mat[2,2]*mat[3,1] - mat[2,1]*mat[3,2]) +
                mat[1,1]*(mat[2,0]*mat[3,2] - mat[2,2]*mat[3,0]) +
                mat[1,2]*(mat[2,1]*mat[3,0] - mat[2,0]*mat[3,1])))

        out_inv[4,0] = (mat[1,0]*(mat[2,1]*(mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2]) +
                mat[2,2]*(mat[3,3]*mat[4,1] - mat[3,1]*mat[4,3]) +
                mat[2,3]*(mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1])) +
                mat[1,1]*(mat[2,0]*(mat[3,3]*mat[4,2] - mat[3,2]*mat[4,3]) +
                mat[2,2]*(mat[3,0]*mat[4,3] - mat[3,3]*mat[4,0]) +
                mat[2,3]*(mat[3,2]*mat[4,0] - mat[3,0]*mat[4,2])) +
                mat[1,2]*(mat[2,0]*(mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1]) +
                mat[2,1]*(mat[3,3]*mat[4,0] - mat[3,0]*mat[4,3]) +
                mat[2,3]*(mat[3,0]*mat[4,1] - mat[3,1]*mat[4,0])) +
                mat[1,3]*(mat[2,0]*(mat[3,2]*mat[4,1] - mat[3,1]*mat[4,2]) +
                mat[2,1]*(mat[3,0]*mat[4,2] - mat[3,2]*mat[4,0]) +
                mat[2,2]*(mat[3,1]*mat[4,0] - mat[3,0]*mat[4,1])))

        out_inv[4,1] = -(mat[0,0]*(mat[2,1]*(mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2]) +
                mat[2,2]*(mat[3,3]*mat[4,1] - mat[3,1]*mat[4,3]) +
                mat[2,3]*(mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1])) +
                mat[0,1]*(mat[2,0]*(mat[3,3]*mat[4,2] - mat[3,2]*mat[4,3]) +
                mat[2,2]*(mat[3,0]*mat[4,3] - mat[3,3]*mat[4,0]) +
                mat[2,3]*(mat[3,2]*mat[4,0] - mat[3,0]*mat[4,2])) +
                mat[0,2]*(mat[2,0]*(mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1]) +
                mat[2,1]*(mat[3,3]*mat[4,0] - mat[3,0]*mat[4,3]) +
                mat[2,3]*(mat[3,0]*mat[4,1] - mat[3,1]*mat[4,0])) +
                mat[0,3]*(mat[2,0]*(mat[3,2]*mat[4,1] - mat[3,1]*mat[4,2]) +
                mat[2,1]*(mat[3,0]*mat[4,2] - mat[3,2]*mat[4,0]) +
                mat[2,2]*(mat[3,1]*mat[4,0] - mat[3,0]*mat[4,1])))

        out_inv[4,2] = (mat[0,0]*(mat[1,1]*(mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2]) +
                mat[1,2]*(mat[3,3]*mat[4,1] - mat[3,1]*mat[4,3]) +
                mat[1,3]*(mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1])) +
                mat[0,1]*(mat[1,0]*(mat[3,3]*mat[4,2] - mat[3,2]*mat[4,3]) +
                mat[1,2]*(mat[3,0]*mat[4,3] - mat[3,3]*mat[4,0]) +
                mat[1,3]*(mat[3,2]*mat[4,0] - mat[3,0]*mat[4,2])) +
                mat[0,2]*(mat[1,0]*(mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1]) +
                mat[1,1]*(mat[3,3]*mat[4,0] - mat[3,0]*mat[4,3]) +
                mat[1,3]*(mat[3,0]*mat[4,1] - mat[3,1]*mat[4,0])) +
                mat[0,3]*(mat[1,0]*(mat[3,2]*mat[4,1] - mat[3,1]*mat[4,2]) +
                mat[1,1]*(mat[3,0]*mat[4,2] - mat[3,2]*mat[4,0]) +
                mat[1,2]*(mat[3,1]*mat[4,0] - mat[3,0]*mat[4,1])))

        out_inv[4,3] = -(mat[0,0]*(mat[1,1]*(mat[2,2]*mat[4,3] - mat[2,3]*mat[4,2]) +
                mat[1,2]*(mat[2,3]*mat[4,1] - mat[2,1]*mat[4,3]) +
                mat[1,3]*(mat[2,1]*mat[4,2] - mat[2,2]*mat[4,1])) +
                mat[0,1]*(mat[1,0]*(mat[2,3]*mat[4,2] - mat[2,2]*mat[4,3]) +
                mat[1,2]*(mat[2,0]*mat[4,3] - mat[2,3]*mat[4,0]) +
                mat[1,3]*(mat[2,2]*mat[4,0] - mat[2,0]*mat[4,2])) +
                mat[0,2]*(mat[1,0]*(mat[2,1]*mat[4,3] - mat[2,3]*mat[4,1]) +
                mat[1,1]*(mat[2,3]*mat[4,0] - mat[2,0]*mat[4,3]) +
                mat[1,3]*(mat[2,0]*mat[4,1] - mat[2,1]*mat[4,0])) +
                mat[0,3]*(mat[1,0]*(mat[2,2]*mat[4,1] - mat[2,1]*mat[4,2]) +
                mat[1,1]*(mat[2,0]*mat[4,2] - mat[2,2]*mat[4,0]) +
                mat[1,2]*(mat[2,1]*mat[4,0] - mat[2,0]*mat[4,1])))

        out_inv[4,4] = (mat[0,0]*(mat[1,1]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                mat[1,2]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3]) +
                mat[1,3]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1])) +
                mat[0,1]*(mat[1,0]*(mat[2,3]*mat[3,2] - mat[2,2]*mat[3,3]) +
                mat[1,2]*(mat[2,0]*mat[3,3] - mat[2,3]*mat[3,0]) +
                mat[1,3]*(mat[2,2]*mat[3,0] - mat[2,0]*mat[3,2])) +
                mat[0,2]*(mat[1,0]*(mat[2,1]*mat[3,3] - mat[2,3]*mat[3,1]) +
                mat[1,1]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3]) +
                mat[1,3]*(mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0])) +
                mat[0,3]*(mat[1,0]*(mat[2,2]*mat[3,1] - mat[2,1]*mat[3,2]) +
                mat[1,1]*(mat[2,0]*mat[3,2] - mat[2,2]*mat[3,0]) +
                mat[1,2]*(mat[2,1]*mat[3,0] - mat[2,0]*mat[3,1])))

        inv_det = 1.0/(mat[0,0]*out_inv[0,0] + mat[1,0]*out_inv[0,1] + mat[2,0]*out_inv[0,2] + mat[3,0]*out_inv[0,3] + mat[4,0]*out_inv[0,4])

        for i in range(k):
            for j in range(k):
                out_inv[i,j] *= inv_det
    else:
        inverse_dgetr(mat, out_inv)


cdef void inverse_symmetric_psd(double[:,:] mat, double[:,:] out_inv):
    """
    Computes the inverse of a symmetric matrix `mat` in a matrix `out_inv`.
    
    **IMPORTANT NOTE: mat should be different to out_inv**.
    """
    cdef Py_ssize_t k = mat.shape[0], i, j
    cdef double inv_det
    # TODO: Check out_inv is different pointer to mat.
    if k == 2:
        inv_det = 1.0 / (mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0])
        out_inv[0,0] = mat[1,1] * inv_det
        out_inv[0,1] = out_inv[1,0] = -mat[0,1] * inv_det
        out_inv[1,1] = mat[0,0] * inv_det
    elif k == 3:
        inv_det = 1.0 / (mat[0,0]*(mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) +
                         mat[0,1]*(mat[1,2]*mat[2,0] - mat[1,0]*mat[2,2]) +
                         mat[0,2]*(mat[1,0]*mat[2,1] - mat[1,1]*mat[2,0]) )

        out_inv[0,0] = (mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) * inv_det
        out_inv[0,1] = out_inv[1,0] =  -(mat[0,1]*mat[2,2] - mat[0,2]*mat[2,1]) * inv_det
        out_inv[0,2] = out_inv[2,0] = (mat[0,1]*mat[1,2] - mat[0,2]*mat[1,1]) * inv_det
        out_inv[1,1] = (mat[0,0]*mat[2,2] - mat[0,2]*mat[2,0]) * inv_det
        out_inv[1,2] = out_inv[2,1] = -(mat[0,0]*mat[1,2] - mat[0,2]*mat[1,0]) * inv_det
        out_inv[2,2] = (mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0]) * inv_det
    elif k == 4:
        out_inv[0,0] = (mat[1,1]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[1,2]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3]) +
                        mat[1,3]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]) )

        out_inv[0,1] = -(mat[0,1]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[0,2]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3]) +
                        mat[0,3]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]) )

        out_inv[0,2] = (mat[0,1]*(mat[1,2]*mat[3,3] - mat[1,3]*mat[3,2]) +
                        mat[0,2]*(mat[1,3]*mat[3,1] - mat[1,1]*mat[3,3]) +
                        mat[0,3]*(mat[1,1]*mat[3,2] - mat[1,2]*mat[3,1]) )

        out_inv[0,3] = -(mat[0,1]*(mat[1,2]*mat[2,3] - mat[1,3]*mat[2,2]) +
                        mat[0,2]*(mat[1,3]*mat[2,1] - mat[1,1]*mat[2,3]) +
                        mat[0,3]*(mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) )

        out_inv[1,1] = (mat[0,0]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[0,2]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3]) +
                        mat[0,3]*(mat[2,0]*mat[3,2] - mat[2,2]*mat[3,0]) )

        out_inv[1,2] = -(mat[0,0]*(mat[1,2]*mat[3,3] - mat[1,3]*mat[3,2]) +
                        mat[0,2]*(mat[1,3]*mat[3,0] - mat[1,0]*mat[3,3]) +
                        mat[0,3]*(mat[1,0]*mat[3,2] - mat[1,2]*mat[3,0]) )

        out_inv[1,3] = (mat[0,0]*(mat[1,2]*mat[2,3] - mat[1,3]*mat[2,2]) +
                        mat[0,2]*(mat[1,3]*mat[2,0] - mat[1,0]*mat[2,3]) +
                        mat[0,3]*(mat[1,0]*mat[2,2] - mat[1,2]*mat[2,0]) )

        out_inv[2,2] = (mat[0,0]*(mat[1,1]*mat[3,3] - mat[1,3]*mat[3,1]) +
                        mat[0,1]*(mat[1,3]*mat[3,0] - mat[1,0]*mat[3,3]) +
                        mat[0,3]*(mat[1,0]*mat[3,1] - mat[1,1]*mat[3,0]) )

        out_inv[2,3] = -(mat[0,0]*(mat[1,1]*mat[2,3] - mat[1,3]*mat[2,1]) +
                         mat[0,1]*(mat[1,3]*mat[2,0] - mat[1,0]*mat[2,3]) +
                         mat[0,3]*(mat[1,0]*mat[2,1] - mat[1,1]*mat[2,0]) )

        out_inv[3,3] = (mat[0,0]*(mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) +
                        mat[0,1]*(mat[1,2]*mat[2,0] - mat[1,0]*mat[2,2]) +
                        mat[0,2]*(mat[1,0]*mat[2,1] - mat[1,1]*mat[2,0]) )

        inv_det = 1.0/(mat[0,0]*out_inv[0,0] + mat[1,0]*out_inv[0,1] + mat[2,0]*out_inv[0,2] + mat[3,0]*out_inv[0,3])

        for i in range(k):
            for j in range(i, k):
                out_inv[i,j] *= inv_det

        out_inv[1,0] = out_inv[0,1]
        out_inv[2,0] = out_inv[0,2]
        out_inv[3,0] = out_inv[0,3]
        out_inv[2,1] = out_inv[1,2]
        out_inv[3,1] = out_inv[1,3]
        out_inv[3,2] = out_inv[2,3]

    elif k==5:
        out_inv[0,0] = (mat[1,1]*(mat[2,2]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[2,3]*(mat[3,4]*mat[4,2] - mat[3,2]*mat[4,4]) +
                mat[2,4]*(mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2])) +
                mat[1,2]*(mat[2,1]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[2,3]*(mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]) +
                mat[2,4]*(mat[3,3]*mat[4,1] - mat[3,1]*mat[4,3])) +
                mat[1,3]*(mat[2,1]*(mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]) +
                mat[2,2]*(mat[3,4]*mat[4,1] - mat[3,1]*mat[4,4]) +
                mat[2,4]*(mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1])) +
                mat[1,4]*(mat[2,1]*(mat[3,3]*mat[4,2] - mat[3,2]*mat[4,3]) +
                mat[2,2]*(mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1]) +
                mat[2,3]*(mat[3,2]*mat[4,1] - mat[3,1]*mat[4,2])))

        out_inv[0,1] = -(mat[0,1]*(mat[2,2]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[2,3]*(mat[3,4]*mat[4,2] - mat[3,2]*mat[4,4]) +
                mat[2,4]*(mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2])) +
                mat[0,2]*(mat[2,1]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[2,3]*(mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]) +
                mat[2,4]*(mat[3,3]*mat[4,1] - mat[3,1]*mat[4,3])) +
                mat[0,3]*(mat[2,1]*(mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]) +
                mat[2,2]*(mat[3,4]*mat[4,1] - mat[3,1]*mat[4,4]) +
                mat[2,4]*(mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1])) +
                mat[0,4]*(mat[2,1]*(mat[3,3]*mat[4,2] - mat[3,2]*mat[4,3]) +
                mat[2,2]*(mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1]) +
                mat[2,3]*(mat[3,2]*mat[4,1] - mat[3,1]*mat[4,2])))

        out_inv[0,2] = (mat[0,1]*(mat[1,2]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[1,3]*(mat[3,4]*mat[4,2] - mat[3,2]*mat[4,4]) +
                mat[1,4]*(mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2])) +
                mat[0,2]*(mat[1,1]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[1,3]*(mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]) +
                mat[1,4]*(mat[3,3]*mat[4,1] - mat[3,1]*mat[4,3])) +
                mat[0,3]*(mat[1,1]*(mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]) +
                mat[1,2]*(mat[3,4]*mat[4,1] - mat[3,1]*mat[4,4]) +
                mat[1,4]*(mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1])) +
                mat[0,4]*(mat[1,1]*(mat[3,3]*mat[4,2] - mat[3,2]*mat[4,3]) +
                mat[1,2]*(mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1]) +
                mat[1,3]*(mat[3,2]*mat[4,1] - mat[3,1]*mat[4,2])))

        out_inv[0,3] = -(mat[0,1]*(mat[1,2]*(mat[2,3]*mat[4,4] - mat[2,4]*mat[4,3]) +
                mat[1,3]*(mat[2,4]*mat[4,2] - mat[2,2]*mat[4,4]) +
                mat[1,4]*(mat[2,2]*mat[4,3] - mat[2,3]*mat[4,2])) +
                mat[0,2]*(mat[1,1]*(mat[2,4]*mat[4,3] - mat[2,3]*mat[4,4]) +
                mat[1,3]*(mat[2,1]*mat[4,4] - mat[2,4]*mat[4,1]) +
                mat[1,4]*(mat[2,3]*mat[4,1] - mat[2,1]*mat[4,3])) +
                mat[0,3]*(mat[1,1]*(mat[2,2]*mat[4,4] - mat[2,4]*mat[4,2]) +
                mat[1,2]*(mat[2,4]*mat[4,1] - mat[2,1]*mat[4,4]) +
                mat[1,4]*(mat[2,1]*mat[4,2] - mat[2,2]*mat[4,1])) +
                mat[0,4]*(mat[1,1]*(mat[2,3]*mat[4,2] - mat[2,2]*mat[4,3]) +
                mat[1,2]*(mat[2,1]*mat[4,3] - mat[2,3]*mat[4,1]) +
                mat[1,3]*(mat[2,2]*mat[4,1] - mat[2,1]*mat[4,2])))

        out_inv[0,4] = (mat[0,1]*(mat[1,2]*(mat[2,3]*mat[3,4] - mat[2,4]*mat[3,3]) +
                mat[1,3]*(mat[2,4]*mat[3,2] - mat[2,2]*mat[3,4]) +
                mat[1,4]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2])) +
                mat[0,2]*(mat[1,1]*(mat[2,4]*mat[3,3] - mat[2,3]*mat[3,4]) +
                mat[1,3]*(mat[2,1]*mat[3,4] - mat[2,4]*mat[3,1]) +
                mat[1,4]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3])) +
                mat[0,3]*(mat[1,1]*(mat[2,2]*mat[3,4] - mat[2,4]*mat[3,2]) +
                mat[1,2]*(mat[2,4]*mat[3,1] - mat[2,1]*mat[3,4]) +
                mat[1,4]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1])) +
                mat[0,4]*(mat[1,1]*(mat[2,3]*mat[3,2] - mat[2,2]*mat[3,3]) +
                mat[1,2]*(mat[2,1]*mat[3,3] - mat[2,3]*mat[3,1]) +
                mat[1,3]*(mat[2,2]*mat[3,1] - mat[2,1]*mat[3,2])))

        out_inv[1,1] = (mat[0,0]*(mat[2,2]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[2,3]*(mat[3,4]*mat[4,2] - mat[3,2]*mat[4,4]) +
                mat[2,4]*(mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2])) +
                mat[0,2]*(mat[2,0]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[2,3]*(mat[3,0]*mat[4,4] - mat[3,4]*mat[4,0]) +
                mat[2,4]*(mat[3,3]*mat[4,0] - mat[3,0]*mat[4,3])) +
                mat[0,3]*(mat[2,0]*(mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]) +
                mat[2,2]*(mat[3,4]*mat[4,0] - mat[3,0]*mat[4,4]) +
                mat[2,4]*(mat[3,0]*mat[4,2] - mat[3,2]*mat[4,0])) +
                mat[0,4]*(mat[2,0]*(mat[3,3]*mat[4,2] - mat[3,2]*mat[4,3]) +
                mat[2,2]*(mat[3,0]*mat[4,3] - mat[3,3]*mat[4,0]) +
                mat[2,3]*(mat[3,2]*mat[4,0] - mat[3,0]*mat[4,2])))

        out_inv[1,2] = -(mat[0,0]*(mat[1,2]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[1,3]*(mat[3,4]*mat[4,2] - mat[3,2]*mat[4,4]) +
                mat[1,4]*(mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2])) +
                mat[0,2]*(mat[1,0]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[1,3]*(mat[3,0]*mat[4,4] - mat[3,4]*mat[4,0]) +
                mat[1,4]*(mat[3,3]*mat[4,0] - mat[3,0]*mat[4,3])) +
                mat[0,3]*(mat[1,0]*(mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]) +
                mat[1,2]*(mat[3,4]*mat[4,0] - mat[3,0]*mat[4,4]) +
                mat[1,4]*(mat[3,0]*mat[4,2] - mat[3,2]*mat[4,0])) +
                mat[0,4]*(mat[1,0]*(mat[3,3]*mat[4,2] - mat[3,2]*mat[4,3]) +
                mat[1,2]*(mat[3,0]*mat[4,3] - mat[3,3]*mat[4,0]) +
                mat[1,3]*(mat[3,2]*mat[4,0] - mat[3,0]*mat[4,2])))

        out_inv[1,3] = (mat[0,0]*(mat[1,2]*(mat[2,3]*mat[4,4] - mat[2,4]*mat[4,3]) +
                mat[1,3]*(mat[2,4]*mat[4,2] - mat[2,2]*mat[4,4]) +
                mat[1,4]*(mat[2,2]*mat[4,3] - mat[2,3]*mat[4,2])) +
                mat[0,2]*(mat[1,0]*(mat[2,4]*mat[4,3] - mat[2,3]*mat[4,4]) +
                mat[1,3]*(mat[2,0]*mat[4,4] - mat[2,4]*mat[4,0]) +
                mat[1,4]*(mat[2,3]*mat[4,0] - mat[2,0]*mat[4,3])) +
                mat[0,3]*(mat[1,0]*(mat[2,2]*mat[4,4] - mat[2,4]*mat[4,2]) +
                mat[1,2]*(mat[2,4]*mat[4,0] - mat[2,0]*mat[4,4]) +
                mat[1,4]*(mat[2,0]*mat[4,2] - mat[2,2]*mat[4,0])) +
                mat[0,4]*(mat[1,0]*(mat[2,3]*mat[4,2] - mat[2,2]*mat[4,3]) +
                mat[1,2]*(mat[2,0]*mat[4,3] - mat[2,3]*mat[4,0]) +
                mat[1,3]*(mat[2,2]*mat[4,0] - mat[2,0]*mat[4,2])))

        out_inv[1,4] = -(mat[0,0]*(mat[1,2]*(mat[2,3]*mat[3,4] - mat[2,4]*mat[3,3]) +
                mat[1,3]*(mat[2,4]*mat[3,2] - mat[2,2]*mat[3,4]) +
                mat[1,4]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2])) +
                mat[0,2]*(mat[1,0]*(mat[2,4]*mat[3,3] - mat[2,3]*mat[3,4]) +
                mat[1,3]*(mat[2,0]*mat[3,4] - mat[2,4]*mat[3,0]) +
                mat[1,4]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3])) +
                mat[0,3]*(mat[1,0]*(mat[2,2]*mat[3,4] - mat[2,4]*mat[3,2]) +
                mat[1,2]*(mat[2,4]*mat[3,0] - mat[2,0]*mat[3,4]) +
                mat[1,4]*(mat[2,0]*mat[3,2] - mat[2,2]*mat[3,0])) +
                mat[0,4]*(mat[1,0]*(mat[2,3]*mat[3,2] - mat[2,2]*mat[3,3]) +
                mat[1,2]*(mat[2,0]*mat[3,3] - mat[2,3]*mat[3,0]) +
                mat[1,3]*(mat[2,2]*mat[3,0] - mat[2,0]*mat[3,2])))

        out_inv[2,2] = (mat[0,0]*(mat[1,1]*(mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]) +
                mat[1,3]*(mat[3,4]*mat[4,1] - mat[3,1]*mat[4,4]) +
                mat[1,4]*(mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1])) +
                mat[0,1]*(mat[1,0]*(mat[3,4]*mat[4,3] - mat[3,3]*mat[4,4]) +
                mat[1,3]*(mat[3,0]*mat[4,4] - mat[3,4]*mat[4,0]) +
                mat[1,4]*(mat[3,3]*mat[4,0] - mat[3,0]*mat[4,3])) +
                mat[0,3]*(mat[1,0]*(mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]) +
                mat[1,1]*(mat[3,4]*mat[4,0] - mat[3,0]*mat[4,4]) +
                mat[1,4]*(mat[3,0]*mat[4,1] - mat[3,1]*mat[4,0])) +
                mat[0,4]*(mat[1,0]*(mat[3,3]*mat[4,1] - mat[3,1]*mat[4,3]) +
                mat[1,1]*(mat[3,0]*mat[4,3] - mat[3,3]*mat[4,0]) +
                mat[1,3]*(mat[3,1]*mat[4,0] - mat[3,0]*mat[4,1])))

        out_inv[2,3] = -(mat[0,0]*(mat[1,1]*(mat[2,3]*mat[4,4] - mat[2,4]*mat[4,3]) +
                mat[1,3]*(mat[2,4]*mat[4,1] - mat[2,1]*mat[4,4]) +
                mat[1,4]*(mat[2,1]*mat[4,3] - mat[2,3]*mat[4,1])) +
                mat[0,1]*(mat[1,0]*(mat[2,4]*mat[4,3] - mat[2,3]*mat[4,4]) +
                mat[1,3]*(mat[2,0]*mat[4,4] - mat[2,4]*mat[4,0]) +
                mat[1,4]*(mat[2,3]*mat[4,0] - mat[2,0]*mat[4,3])) +
                mat[0,3]*(mat[1,0]*(mat[2,1]*mat[4,4] - mat[2,4]*mat[4,1]) +
                mat[1,1]*(mat[2,4]*mat[4,0] - mat[2,0]*mat[4,4]) +
                mat[1,4]*(mat[2,0]*mat[4,1] - mat[2,1]*mat[4,0])) +
                mat[0,4]*(mat[1,0]*(mat[2,3]*mat[4,1] - mat[2,1]*mat[4,3]) +
                mat[1,1]*(mat[2,0]*mat[4,3] - mat[2,3]*mat[4,0]) +
                mat[1,3]*(mat[2,1]*mat[4,0] - mat[2,0]*mat[4,1])))

        out_inv[2,4] = (mat[0,0]*(mat[1,1]*(mat[2,3]*mat[3,4] - mat[2,4]*mat[3,3]) +
                mat[1,3]*(mat[2,4]*mat[3,1] - mat[2,1]*mat[3,4]) +
                mat[1,4]*(mat[2,1]*mat[3,3] - mat[2,3]*mat[3,1])) +
                mat[0,1]*(mat[1,0]*(mat[2,4]*mat[3,3] - mat[2,3]*mat[3,4]) +
                mat[1,3]*(mat[2,0]*mat[3,4] - mat[2,4]*mat[3,0]) +
                mat[1,4]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3])) +
                mat[0,3]*(mat[1,0]*(mat[2,1]*mat[3,4] - mat[2,4]*mat[3,1]) +
                mat[1,1]*(mat[2,4]*mat[3,0] - mat[2,0]*mat[3,4]) +
                mat[1,4]*(mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0])) +
                mat[0,4]*(mat[1,0]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3]) +
                mat[1,1]*(mat[2,0]*mat[3,3] - mat[2,3]*mat[3,0]) +
                mat[1,3]*(mat[2,1]*mat[3,0] - mat[2,0]*mat[3,1])))

        out_inv[3,3] = (mat[0,0]*(mat[1,1]*(mat[2,2]*mat[4,4] - mat[2,4]*mat[4,2]) +
                mat[1,2]*(mat[2,4]*mat[4,1] - mat[2,1]*mat[4,4]) +
                mat[1,4]*(mat[2,1]*mat[4,2] - mat[2,2]*mat[4,1])) +
                mat[0,1]*(mat[1,0]*(mat[2,4]*mat[4,2] - mat[2,2]*mat[4,4]) +
                mat[1,2]*(mat[2,0]*mat[4,4] - mat[2,4]*mat[4,0]) +
                mat[1,4]*(mat[2,2]*mat[4,0] - mat[2,0]*mat[4,2])) +
                mat[0,2]*(mat[1,0]*(mat[2,1]*mat[4,4] - mat[2,4]*mat[4,1]) +
                mat[1,1]*(mat[2,4]*mat[4,0] - mat[2,0]*mat[4,4]) +
                mat[1,4]*(mat[2,0]*mat[4,1] - mat[2,1]*mat[4,0])) +
                mat[0,4]*(mat[1,0]*(mat[2,2]*mat[4,1] - mat[2,1]*mat[4,2]) +
                mat[1,1]*(mat[2,0]*mat[4,2] - mat[2,2]*mat[4,0]) +
                mat[1,2]*(mat[2,1]*mat[4,0] - mat[2,0]*mat[4,1])))

        out_inv[3,4] = -(mat[0,0]*(mat[1,1]*(mat[2,2]*mat[3,4] - mat[2,4]*mat[3,2]) +
                mat[1,2]*(mat[2,4]*mat[3,1] - mat[2,1]*mat[3,4]) +
                mat[1,4]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1])) +
                mat[0,1]*(mat[1,0]*(mat[2,4]*mat[3,2] - mat[2,2]*mat[3,4]) +
                mat[1,2]*(mat[2,0]*mat[3,4] - mat[2,4]*mat[3,0]) +
                mat[1,4]*(mat[2,2]*mat[3,0] - mat[2,0]*mat[3,2])) +
                mat[0,2]*(mat[1,0]*(mat[2,1]*mat[3,4] - mat[2,4]*mat[3,1]) +
                mat[1,1]*(mat[2,4]*mat[3,0] - mat[2,0]*mat[3,4]) +
                mat[1,4]*(mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0])) +
                mat[0,4]*(mat[1,0]*(mat[2,2]*mat[3,1] - mat[2,1]*mat[3,2]) +
                mat[1,1]*(mat[2,0]*mat[3,2] - mat[2,2]*mat[3,0]) +
                mat[1,2]*(mat[2,1]*mat[3,0] - mat[2,0]*mat[3,1])))

        out_inv[4,4] = (mat[0,0]*(mat[1,1]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                mat[1,2]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3]) +
                mat[1,3]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1])) +
                mat[0,1]*(mat[1,0]*(mat[2,3]*mat[3,2] - mat[2,2]*mat[3,3]) +
                mat[1,2]*(mat[2,0]*mat[3,3] - mat[2,3]*mat[3,0]) +
                mat[1,3]*(mat[2,2]*mat[3,0] - mat[2,0]*mat[3,2])) +
                mat[0,2]*(mat[1,0]*(mat[2,1]*mat[3,3] - mat[2,3]*mat[3,1]) +
                mat[1,1]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3]) +
                mat[1,3]*(mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0])) +
                mat[0,3]*(mat[1,0]*(mat[2,2]*mat[3,1] - mat[2,1]*mat[3,2]) +
                mat[1,1]*(mat[2,0]*mat[3,2] - mat[2,2]*mat[3,0]) +
                mat[1,2]*(mat[2,1]*mat[3,0] - mat[2,0]*mat[3,1])))

        inv_det = 1.0/(mat[0,0]*out_inv[0,0] + mat[1,0]*out_inv[0,1] + mat[2,0]*out_inv[0,2] + mat[3,0]*out_inv[0,3] + mat[4,0]*out_inv[0,4])

        for i in range(k):
            for j in range(i,k):
                out_inv[i,j] *= inv_det

        out_inv[1,0] = out_inv[0,1]
        out_inv[2,0] = out_inv[0,2]
        out_inv[3,0] = out_inv[0,3]
        out_inv[4,0] = out_inv[0,4]
        out_inv[2,1] = out_inv[1,2]
        out_inv[3,1] = out_inv[1,3]
        out_inv[4,1] = out_inv[1,4]
        out_inv[3,2] = out_inv[2,3]
        out_inv[4,2] = out_inv[2,4]
        out_inv[4,3] = out_inv[3,4]

    else:
        inverse_dpotr(mat, out_inv)

cdef dgemv(double alpha, double[:,:] A, double[:] x, double beta, double[:] y):
    """
    See http://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_gadd421a107a488d524859b4a64c1901a9.html
    :param alpha: 
    :param A: 
    :param x: 
    :param beta: 
    :param y: 
    :return: 
    """
    cdef char trans = 'N'
    cdef int M = A.shape[0], N = A.shape[1], incx = 1, incy = 1

    cython_blas.dgemv(&trans, &M, &N, &alpha, &A[0,0], &M, &x[0], &incx, &beta, &y[0], &incy)