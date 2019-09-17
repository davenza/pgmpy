#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
from libc.stdlib cimport calloc, malloc, free
from libc.string cimport memcpy

cimport scipy.linalg.cython_lapack as cython_lapack
cimport scipy.linalg.cython_blas as cython_blas

import sys

cdef void inverse_dgetr(double[:,:] mat, double[:,:] out_inv):
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

    cdef int len_workspace = k*k
    workspace = <double*>malloc(sizeof(double)*len_workspace)

                # this is uninitialised (the contents are arbitrary)
                # but that's OK because it's used as an output
    try:
        cython_lapack.dgetrf(&k,&k,mat_pointer,&k,piv_pointer,&info)
        cython_lapack.dgetri(&k,mat_pointer,&k,piv_pointer, workspace, &len_workspace, &info)
        # you should check info to ensure it's worked
    finally:
        free(piv_pointer) # the "try ... finally" ensures that this is freed
        free(workspace) # the "try ... finally" ensures that this is freed


cdef void inverse_dpotr(double[:,:] mat, double[:,:] out_inv):
    """
    This function inverses symmetric positive definite matrices.
    :param mat: 
    :return: 
    """
    # you should probably check that mat.shape[0]==mat.shape[1]
    # and that mat is actually a float64 array
    cdef int k = mat.shape[0]
    cdef char uplo = 'U'
    cdef Py_ssize_t i,j

    if (&mat[0,0] != &out_inv[0,0]):
        memcpy(&out_inv[0,0], &mat[0,0], k*k*sizeof(double))

    cdef int info = 0

    cdef double* mat_pointer = &out_inv[0,0]
    # I suspect you should be doing a check that mat_pointer has been assigned
    cython_lapack.dpotrf(&uplo,&k,mat_pointer,&k,&info)
    cython_lapack.dpotri(&uplo,&k,mat_pointer,&k,&info)

    for i in range(k):
        for j in range(i+1,k):
            out_inv[i,j] = out_inv[j,i]

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

# ######################################################################################################
# ######################################################################################################
# The cached determinants versions used for k=5 and k=6 are generated with the following code. They are called as:
# rows = cols = list(range(5))
# code, cache5 = adjugate_str_cached(rows, cols, "mat", "tmp_inv")
# print(code)

# ######################################################################################################
# ######################################################################################################
# def generate_tmp_name(rows, cols):
#     return "det_" + ''.join(map(str, sorted(rows))) + "_" + ''.join(map(str, sorted(cols)))
#
# def cache_adjugate(rows, cols, positive, in_str):
#     cache = {}
#     if len(rows) == 2 and len(cols) == 2:
#         cache[(generate_tmp_name(rows, cols),2)] = in_str + "[" + str(rows[0]) + "," + str(cols[0]) + "]*" + \
#                 in_str + "[" + str(rows[1]) + "," + str(cols[1]) + "] - " + \
#                    in_str + "[" + str(rows[0]) + "," + str(cols[1]) + "]*" + \
#                    in_str + "[" + str(rows[1]) + "," + str(cols[0]) + "]"
#     else:
#         for i in range(len(cols)):
#             removed_rows = rows.copy()
#             del removed_rows[0]
#             removed_cols = cols.copy()
#             del removed_cols[i]
#
#             if (positive and i % 2 == 0) or ((not positive) and i % 2 == 1):
#                 partial_cache = cache_adjugate(removed_rows, removed_cols, True, in_str)
#                 cache.update(partial_cache)
#
#                 if not (generate_tmp_name(rows, cols), len(rows)) in cache.keys():
#                     cache[(generate_tmp_name(rows, cols), len(rows))] = (in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(" + generate_tmp_name(removed_rows, removed_cols) + ")")
#                 else:
#                     cache[(generate_tmp_name(rows, cols), len(rows))] += (' \\\n\t\t +' + in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(" + generate_tmp_name(removed_rows, removed_cols) + ")")
#             else:
#                 partial_cache = cache_adjugate(removed_rows, removed_cols, True, in_str)
#                 cache.update(partial_cache)
#
#                 if not (generate_tmp_name(rows, cols), len(rows)) in cache.keys():
#                     cache[(generate_tmp_name(rows, cols), len(rows))] = (in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(-" + generate_tmp_name(removed_rows, removed_cols) + ")")
#                 else:
#                     cache[(generate_tmp_name(rows, cols), len(rows))] += (' \\\n\t\t +' + in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(-" + generate_tmp_name(removed_rows, removed_cols) + ")")
#
#     return cache
#
# def det_str_cached(rows, cols, positive, in_str):
#     if len(rows) == 2 and len(cols) == 2:
#         return generate_tmp_name(rows, cols)
#     else:
#         output_str = ""
#         for i in range(len(cols)):
#             removed_rows = rows.copy()
#             del removed_rows[0]
#             removed_cols = cols.copy()
#             del removed_cols[i]
#
#             if (positive and i % 2 == 0) or ((not positive) and i % 2 == 1):
#                 if output_str == "":
#                     output_str += (in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(" + generate_tmp_name(removed_rows, removed_cols) + ")")
#                 else:
#                     output_str += (" \\\n\t\t + " + in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(" + generate_tmp_name(removed_rows, removed_cols) + ")")
#             else:
#                 if output_str == "":
#                     output_str += (in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(-" + generate_tmp_name(removed_rows, removed_cols) + ")")
#                 else:
#                     output_str += (" \\\n\t\t + " + in_str + "[" + str(rows[0]) + "," + str(cols[i]) + "]*(-" + generate_tmp_name(removed_rows, removed_cols) + ")")
#
#
#     return output_str
#
# def adjugate_str_cached(rows, cols, in_str, out_str):
#     cache = {}
#
#     res = ""
#     # for j in range(len(cols)):
#     for j in range(1):
#         for i in range(len(rows)):
#             removed_rows = rows.copy()
#             del removed_rows[i]
#             removed_cols = cols.copy()
#             del removed_cols[j]
#             if ((i + j) % 2) == 0:
#                 partial_cache = cache_adjugate(removed_rows, removed_cols, True, in_str)
#                 cache.update(partial_cache)
#             else:
#                 partial_cache = cache_adjugate(removed_rows, removed_cols, True, in_str)
#                 cache.update(partial_cache)
#
#     sorted_keys = sorted(cache.keys(), key=lambda kv: (kv[1], kv[0]))
#
#     res += "cdef double " + sorted_keys[0][0]
#     for key in sorted_keys[1:]:
#         if key[1] < len(rows) - 1:
#             res += "," + key[0]
#
#     res += '\n\n'
#
#     for key in sorted_keys:
#         if key[1] < len(rows) - 1:
#             res += key[0] + " = " + cache[key] + '\n'
#
#     res += '\n\n'
#     # for j in range(len(cols)):
#     for j in range(1):
#         for i in range(len(rows)):
#             removed_rows = rows.copy()
#             del removed_rows[i]
#             removed_cols = cols.copy()
#             del removed_cols[j]
#             if ((i+j) % 2) == 0:
#                 res += out_str + "[" + str(j) + "][" + str(i) + "] = " + det_str_cached(removed_rows, removed_cols, True, in_str) + "\n"
#             else:
#                 res += out_str + "[" + str(j) + "][" + str(i) + "] = " + det_str_cached(removed_rows, removed_cols, False, in_str) + "\n"
#
#             res += '\n'
#
#     return res, cache



cdef void inverse(double[:,:] mat, double[:,:] out_inv):
    """
    Computes the inverse of a matrix `mat` in a matrix `out_inv`.
    
    **IMPORTANT NOTE: mat should be different to out_inv**.
    """
    cdef Py_ssize_t k = mat.shape[0], i, j
    cdef double inv_det
    cdef double tmp_inv[5][5]
    cdef double det_23_01,det_23_02,det_23_03,det_23_04,det_23_12,det_23_13,det_23_14,det_23_23,det_23_24,det_23_34,\
        det_24_01,det_24_02,det_24_03,det_24_04,det_24_12,det_24_13,det_24_14,det_24_23,det_24_24,det_24_34,det_34_01,\
        det_34_02,det_34_03,det_34_04,det_34_12,det_34_13,det_34_14,det_34_23,det_34_24,det_34_34,det_123_012,\
        det_123_013,det_123_014,det_123_023,det_123_024,det_123_034,det_123_123,det_123_124,det_123_134,det_123_234,\
        det_124_012,det_124_013,det_124_014,det_124_023,det_124_024,det_124_034,det_124_123,det_124_124,det_124_134,\
        det_124_234,det_134_012,det_134_013,det_134_014,det_134_023,det_134_024,det_134_034,det_134_123,det_134_124,\
        det_134_134,det_134_234,det_234_012,det_234_013,det_234_014,det_234_023,det_234_024,det_234_034,det_234_123,\
        det_234_124,det_234_134,det_234_234

    if k == 2:
        inv_det = 1.0 / (mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0])
        tmp_inv[0][0] = mat[0,0]
        out_inv[0,0] = mat[1,1] * inv_det
        out_inv[0,1] = -mat[0,1] * inv_det
        out_inv[1,0] = -mat[1,0] * inv_det
        out_inv[1,1] = tmp_inv[0][0] * inv_det
    elif k == 3:
        inv_det = 1.0 / (mat[0,0]*(mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) +
                         mat[0,1]*(mat[1,2]*mat[2,0] - mat[1,0]*mat[2,2]) +
                         mat[0,2]*(mat[1,0]*mat[2,1] - mat[1,1]*mat[2,0]) )

        tmp_inv[0][0] = (mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1])
        tmp_inv[0][1] = -(mat[0,1]*mat[2,2] - mat[0,2]*mat[2,1])
        tmp_inv[0][2] = (mat[0,1]*mat[1,2] - mat[0,2]*mat[1,1])
        tmp_inv[1][0] = -(mat[1,0]*mat[2,2] - mat[1,2]*mat[2,0])
        tmp_inv[1][1] = (mat[0,0]*mat[2,2] - mat[0,2]*mat[2,0])
        tmp_inv[1][2] = -(mat[0,0]*mat[1,2] - mat[0,2]*mat[1,0])
        tmp_inv[2][0] = (mat[1,0]*mat[2,1] - mat[1,1]*mat[2,0])
        tmp_inv[2][1] = -(mat[0,0]*mat[2,1] - mat[0,1]*mat[2,0])
        tmp_inv[2][2] = (mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0])

        for i in range(k):
            for j in range(k):
                out_inv[i,j] = tmp_inv[i][j] * inv_det

    elif k == 4:
        tmp_inv[0][0] = (mat[1,1]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[1,2]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3]) +
                        mat[1,3]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]) )

        tmp_inv[0][1] = -(mat[0,1]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[0,2]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3]) +
                        mat[0,3]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]) )

        tmp_inv[0][2] = (mat[0,1]*(mat[1,2]*mat[3,3] - mat[1,3]*mat[3,2]) +
                        mat[0,2]*(mat[1,3]*mat[3,1] - mat[1,1]*mat[3,3]) +
                        mat[0,3]*(mat[1,1]*mat[3,2] - mat[1,2]*mat[3,1]) )

        tmp_inv[0][3] = -(mat[0,1]*(mat[1,2]*mat[2,3] - mat[1,3]*mat[2,2]) +
                        mat[0,2]*(mat[1,3]*mat[2,1] - mat[1,1]*mat[2,3]) +
                        mat[0,3]*(mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) )

        tmp_inv[1][0] = -(mat[1,0]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[1,2]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3]) +
                        mat[1,3]*(mat[2,0]*mat[3,2] - mat[2,2]*mat[3,0]) )

        tmp_inv[1][1] = (mat[0,0]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[0,2]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3]) +
                        mat[0,3]*(mat[2,0]*mat[3,2] - mat[2,2]*mat[3,0]) )

        tmp_inv[1][2] = -(mat[0,0]*(mat[1,2]*mat[3,3] - mat[1,3]*mat[3,2]) +
                        mat[0,2]*(mat[1,3]*mat[3,0] - mat[1,0]*mat[3,3]) +
                        mat[0,3]*(mat[1,0]*mat[3,2] - mat[1,2]*mat[3,0]) )

        tmp_inv[1][3] = (mat[0,0]*(mat[1,2]*mat[2,3] - mat[1,3]*mat[2,2]) +
                        mat[0,2]*(mat[1,3]*mat[2,0] - mat[1,0]*mat[2,3]) +
                        mat[0,3]*(mat[1,0]*mat[2,2] - mat[1,2]*mat[2,0]) )

        tmp_inv[2][0] = (mat[1,0]*(mat[2,1]*mat[3,3] - mat[2,3]*mat[3,1]) +
                        mat[1,1]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3]) +
                        mat[1,3]*(mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0]) )

        tmp_inv[2][1] = -(mat[0,0]*(mat[2,1]*mat[3,3] - mat[2,3]*mat[3,1]) +
                        mat[0,1]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3]) +
                        mat[0,3]*(mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0]) )

        tmp_inv[2][2] = (mat[0,0]*(mat[1,1]*mat[3,3] - mat[1,3]*mat[3,1]) +
                        mat[0,1]*(mat[1,3]*mat[3,0] - mat[1,0]*mat[3,3]) +
                        mat[0,3]*(mat[1,0]*mat[3,1] - mat[1,1]*mat[3,0]) )

        tmp_inv[2][3] = -(mat[0,0]*(mat[1,1]*mat[2,3] - mat[1,3]*mat[2,1]) +
                         mat[0,1]*(mat[1,3]*mat[2,0] - mat[1,0]*mat[2,3]) +
                         mat[0,3]*(mat[1,0]*mat[2,1] - mat[1,1]*mat[2,0]) )

        tmp_inv[3][0] = -(mat[1,0]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]) +
                         mat[1,1]*(mat[2,2]*mat[3,0] - mat[2,0]*mat[3,2]) +
                         mat[1,2]*(mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0]) )

        tmp_inv[3][1] = (mat[0,0]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]) +
                        mat[0,1]*(mat[2,2]*mat[3,0] - mat[2,0]*mat[3,2]) +
                        mat[0,2]*(mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0]) )

        tmp_inv[3][2] = -(mat[0,0]*(mat[1,1]*mat[3,2] - mat[1,2]*mat[3,1]) +
                         mat[0,1]*(mat[1,2]*mat[3,0] - mat[1,0]*mat[3,2]) +
                         mat[0,2]*(mat[1,0]*mat[3,1] - mat[1,1]*mat[3,0]) )

        tmp_inv[3][3] = (mat[0,0]*(mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) +
                        mat[0,1]*(mat[1,2]*mat[2,0] - mat[1,0]*mat[2,2]) +
                        mat[0,2]*(mat[1,0]*mat[2,1] - mat[1,1]*mat[2,0]) )

        inv_det = 1.0/(mat[0,0]*tmp_inv[0][0] + mat[1,0]*tmp_inv[0][1] + mat[2,0]*tmp_inv[0][2] + mat[3,0]*tmp_inv[0][3])

        for i in range(k):
            for j in range(k):
                out_inv[i,j] = tmp_inv[i][j] * inv_det

    elif k==5:
        det_23_01 = mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0]
        det_23_02 = mat[2,0]*mat[3,2] - mat[2,2]*mat[3,0]
        det_23_03 = mat[2,0]*mat[3,3] - mat[2,3]*mat[3,0]
        det_23_04 = mat[2,0]*mat[3,4] - mat[2,4]*mat[3,0]
        det_23_12 = mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]
        det_23_13 = mat[2,1]*mat[3,3] - mat[2,3]*mat[3,1]
        det_23_14 = mat[2,1]*mat[3,4] - mat[2,4]*mat[3,1]
        det_23_23 = mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]
        det_23_24 = mat[2,2]*mat[3,4] - mat[2,4]*mat[3,2]
        det_23_34 = mat[2,3]*mat[3,4] - mat[2,4]*mat[3,3]
        det_24_01 = mat[2,0]*mat[4,1] - mat[2,1]*mat[4,0]
        det_24_02 = mat[2,0]*mat[4,2] - mat[2,2]*mat[4,0]
        det_24_03 = mat[2,0]*mat[4,3] - mat[2,3]*mat[4,0]
        det_24_04 = mat[2,0]*mat[4,4] - mat[2,4]*mat[4,0]
        det_24_12 = mat[2,1]*mat[4,2] - mat[2,2]*mat[4,1]
        det_24_13 = mat[2,1]*mat[4,3] - mat[2,3]*mat[4,1]
        det_24_14 = mat[2,1]*mat[4,4] - mat[2,4]*mat[4,1]
        det_24_23 = mat[2,2]*mat[4,3] - mat[2,3]*mat[4,2]
        det_24_24 = mat[2,2]*mat[4,4] - mat[2,4]*mat[4,2]
        det_24_34 = mat[2,3]*mat[4,4] - mat[2,4]*mat[4,3]
        det_34_01 = mat[3,0]*mat[4,1] - mat[3,1]*mat[4,0]
        det_34_02 = mat[3,0]*mat[4,2] - mat[3,2]*mat[4,0]
        det_34_03 = mat[3,0]*mat[4,3] - mat[3,3]*mat[4,0]
        det_34_04 = mat[3,0]*mat[4,4] - mat[3,4]*mat[4,0]
        det_34_12 = mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1]
        det_34_13 = mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1]
        det_34_14 = mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]
        det_34_23 = mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2]
        det_34_24 = mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]
        det_34_34 = mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]
        det_123_012 = mat[1,0]*(det_23_12) \
                 +mat[1,1]*(-det_23_02) \
                 +mat[1,2]*(det_23_01)
        det_123_013 = mat[1,0]*(det_23_13) \
                 +mat[1,1]*(-det_23_03) \
                 +mat[1,3]*(det_23_01)
        det_123_014 = mat[1,0]*(det_23_14) \
                 +mat[1,1]*(-det_23_04) \
                 +mat[1,4]*(det_23_01)
        det_123_023 = mat[1,0]*(det_23_23) \
                 +mat[1,2]*(-det_23_03) \
                 +mat[1,3]*(det_23_02)
        det_123_024 = mat[1,0]*(det_23_24) \
                 +mat[1,2]*(-det_23_04) \
                 +mat[1,4]*(det_23_02)
        det_123_034 = mat[1,0]*(det_23_34) \
                 +mat[1,3]*(-det_23_04) \
                 +mat[1,4]*(det_23_03)
        det_123_123 = mat[1,1]*(det_23_23) \
                 +mat[1,2]*(-det_23_13) \
                 +mat[1,3]*(det_23_12)
        det_123_124 = mat[1,1]*(det_23_24) \
                 +mat[1,2]*(-det_23_14) \
                 +mat[1,4]*(det_23_12)
        det_123_134 = mat[1,1]*(det_23_34) \
                 +mat[1,3]*(-det_23_14) \
                 +mat[1,4]*(det_23_13)
        det_123_234 = mat[1,2]*(det_23_34) \
                 +mat[1,3]*(-det_23_24) \
                 +mat[1,4]*(det_23_23)
        det_124_012 = mat[1,0]*(det_24_12) \
                 +mat[1,1]*(-det_24_02) \
                 +mat[1,2]*(det_24_01)
        det_124_013 = mat[1,0]*(det_24_13) \
                 +mat[1,1]*(-det_24_03) \
                 +mat[1,3]*(det_24_01)
        det_124_014 = mat[1,0]*(det_24_14) \
                 +mat[1,1]*(-det_24_04) \
                 +mat[1,4]*(det_24_01)
        det_124_023 = mat[1,0]*(det_24_23) \
                 +mat[1,2]*(-det_24_03) \
                 +mat[1,3]*(det_24_02)
        det_124_024 = mat[1,0]*(det_24_24) \
                 +mat[1,2]*(-det_24_04) \
                 +mat[1,4]*(det_24_02)
        det_124_034 = mat[1,0]*(det_24_34) \
                 +mat[1,3]*(-det_24_04) \
                 +mat[1,4]*(det_24_03)
        det_124_123 = mat[1,1]*(det_24_23) \
                 +mat[1,2]*(-det_24_13) \
                 +mat[1,3]*(det_24_12)
        det_124_124 = mat[1,1]*(det_24_24) \
                 +mat[1,2]*(-det_24_14) \
                 +mat[1,4]*(det_24_12)
        det_124_134 = mat[1,1]*(det_24_34) \
                 +mat[1,3]*(-det_24_14) \
                 +mat[1,4]*(det_24_13)
        det_124_234 = mat[1,2]*(det_24_34) \
                 +mat[1,3]*(-det_24_24) \
                 +mat[1,4]*(det_24_23)
        det_134_012 = mat[1,0]*(det_34_12) \
                 +mat[1,1]*(-det_34_02) \
                 +mat[1,2]*(det_34_01)
        det_134_013 = mat[1,0]*(det_34_13) \
                 +mat[1,1]*(-det_34_03) \
                 +mat[1,3]*(det_34_01)
        det_134_014 = mat[1,0]*(det_34_14) \
                 +mat[1,1]*(-det_34_04) \
                 +mat[1,4]*(det_34_01)
        det_134_023 = mat[1,0]*(det_34_23) \
                 +mat[1,2]*(-det_34_03) \
                 +mat[1,3]*(det_34_02)
        det_134_024 = mat[1,0]*(det_34_24) \
                 +mat[1,2]*(-det_34_04) \
                 +mat[1,4]*(det_34_02)
        det_134_034 = mat[1,0]*(det_34_34) \
                 +mat[1,3]*(-det_34_04) \
                 +mat[1,4]*(det_34_03)
        det_134_123 = mat[1,1]*(det_34_23) \
                 +mat[1,2]*(-det_34_13) \
                 +mat[1,3]*(det_34_12)
        det_134_124 = mat[1,1]*(det_34_24) \
                 +mat[1,2]*(-det_34_14) \
                 +mat[1,4]*(det_34_12)
        det_134_134 = mat[1,1]*(det_34_34) \
                 +mat[1,3]*(-det_34_14) \
                 +mat[1,4]*(det_34_13)
        det_134_234 = mat[1,2]*(det_34_34) \
                 +mat[1,3]*(-det_34_24) \
                 +mat[1,4]*(det_34_23)
        det_234_012 = mat[2,0]*(det_34_12) \
                 +mat[2,1]*(-det_34_02) \
                 +mat[2,2]*(det_34_01)
        det_234_013 = mat[2,0]*(det_34_13) \
                 +mat[2,1]*(-det_34_03) \
                 +mat[2,3]*(det_34_01)
        det_234_014 = mat[2,0]*(det_34_14) \
                 +mat[2,1]*(-det_34_04) \
                 +mat[2,4]*(det_34_01)
        det_234_023 = mat[2,0]*(det_34_23) \
                 +mat[2,2]*(-det_34_03) \
                 +mat[2,3]*(det_34_02)
        det_234_024 = mat[2,0]*(det_34_24) \
                 +mat[2,2]*(-det_34_04) \
                 +mat[2,4]*(det_34_02)
        det_234_034 = mat[2,0]*(det_34_34) \
                 +mat[2,3]*(-det_34_04) \
                 +mat[2,4]*(det_34_03)
        det_234_123 = mat[2,1]*(det_34_23) \
                 +mat[2,2]*(-det_34_13) \
                 +mat[2,3]*(det_34_12)
        det_234_124 = mat[2,1]*(det_34_24) \
                 +mat[2,2]*(-det_34_14) \
                 +mat[2,4]*(det_34_12)
        det_234_134 = mat[2,1]*(det_34_34) \
                 +mat[2,3]*(-det_34_14) \
                 +mat[2,4]*(det_34_13)
        det_234_234 = mat[2,2]*(det_34_34) \
                 +mat[2,3]*(-det_34_24) \
                 +mat[2,4]*(det_34_23)

        tmp_inv[0][0] = mat[1,1]*(det_234_234) \
                 + mat[1,2]*(-det_234_134) \
                 + mat[1,3]*(det_234_124) \
                 + mat[1,4]*(-det_234_123)

        tmp_inv[1][0] = mat[1,0]*(-det_234_234) \
                 + mat[1,2]*(det_234_034) \
                 + mat[1,3]*(-det_234_024) \
                 + mat[1,4]*(det_234_023)

        tmp_inv[2][0] = mat[1,0]*(det_234_134) \
                 + mat[1,1]*(-det_234_034) \
                 + mat[1,3]*(det_234_014) \
                 + mat[1,4]*(-det_234_013)

        tmp_inv[3][0] = mat[1,0]*(-det_234_124) \
                 + mat[1,1]*(det_234_024) \
                 + mat[1,2]*(-det_234_014) \
                 + mat[1,4]*(det_234_012)

        tmp_inv[4][0] = mat[1,0]*(det_234_123) \
                 + mat[1,1]*(-det_234_023) \
                 + mat[1,2]*(det_234_013) \
                 + mat[1,3]*(-det_234_012)

        tmp_inv[0][1] = mat[0,1]*(-det_234_234) \
                 + mat[0,2]*(det_234_134) \
                 + mat[0,3]*(-det_234_124) \
                 + mat[0,4]*(det_234_123)

        tmp_inv[1][1] = mat[0,0]*(det_234_234) \
                 + mat[0,2]*(-det_234_034) \
                 + mat[0,3]*(det_234_024) \
                 + mat[0,4]*(-det_234_023)

        tmp_inv[2][1] = mat[0,0]*(-det_234_134) \
                 + mat[0,1]*(det_234_034) \
                 + mat[0,3]*(-det_234_014) \
                 + mat[0,4]*(det_234_013)

        tmp_inv[3][1] = mat[0,0]*(det_234_124) \
                 + mat[0,1]*(-det_234_024) \
                 + mat[0,2]*(det_234_014) \
                 + mat[0,4]*(-det_234_012)

        tmp_inv[4][1] = mat[0,0]*(-det_234_123) \
                 + mat[0,1]*(det_234_023) \
                 + mat[0,2]*(-det_234_013) \
                 + mat[0,3]*(det_234_012)

        tmp_inv[0][2] = mat[0,1]*(det_134_234) \
                 + mat[0,2]*(-det_134_134) \
                 + mat[0,3]*(det_134_124) \
                 + mat[0,4]*(-det_134_123)

        tmp_inv[1][2] = mat[0,0]*(-det_134_234) \
                 + mat[0,2]*(det_134_034) \
                 + mat[0,3]*(-det_134_024) \
                 + mat[0,4]*(det_134_023)

        tmp_inv[2][2] = mat[0,0]*(det_134_134) \
                 + mat[0,1]*(-det_134_034) \
                 + mat[0,3]*(det_134_014) \
                 + mat[0,4]*(-det_134_013)

        tmp_inv[3][2] = mat[0,0]*(-det_134_124) \
                 + mat[0,1]*(det_134_024) \
                 + mat[0,2]*(-det_134_014) \
                 + mat[0,4]*(det_134_012)

        tmp_inv[4][2] = mat[0,0]*(det_134_123) \
                 + mat[0,1]*(-det_134_023) \
                 + mat[0,2]*(det_134_013) \
                 + mat[0,3]*(-det_134_012)

        tmp_inv[0][3] = mat[0,1]*(-det_124_234) \
                 + mat[0,2]*(det_124_134) \
                 + mat[0,3]*(-det_124_124) \
                 + mat[0,4]*(det_124_123)

        tmp_inv[1][3] = mat[0,0]*(det_124_234) \
                 + mat[0,2]*(-det_124_034) \
                 + mat[0,3]*(det_124_024) \
                 + mat[0,4]*(-det_124_023)

        tmp_inv[2][3] = mat[0,0]*(-det_124_134) \
                 + mat[0,1]*(det_124_034) \
                 + mat[0,3]*(-det_124_014) \
                 + mat[0,4]*(det_124_013)

        tmp_inv[3][3] = mat[0,0]*(det_124_124) \
                 + mat[0,1]*(-det_124_024) \
                 + mat[0,2]*(det_124_014) \
                 + mat[0,4]*(-det_124_012)

        tmp_inv[4][3] = mat[0,0]*(-det_124_123) \
                 + mat[0,1]*(det_124_023) \
                 + mat[0,2]*(-det_124_013) \
                 + mat[0,3]*(det_124_012)

        tmp_inv[0][4] = mat[0,1]*(det_123_234) \
                 + mat[0,2]*(-det_123_134) \
                 + mat[0,3]*(det_123_124) \
                 + mat[0,4]*(-det_123_123)

        tmp_inv[1][4] = mat[0,0]*(-det_123_234) \
                 + mat[0,2]*(det_123_034) \
                 + mat[0,3]*(-det_123_024) \
                 + mat[0,4]*(det_123_023)

        tmp_inv[2][4] = mat[0,0]*(det_123_134) \
                 + mat[0,1]*(-det_123_034) \
                 + mat[0,3]*(det_123_014) \
                 + mat[0,4]*(-det_123_013)

        tmp_inv[3][4] = mat[0,0]*(-det_123_124) \
                 + mat[0,1]*(det_123_024) \
                 + mat[0,2]*(-det_123_014) \
                 + mat[0,4]*(det_123_012)

        tmp_inv[4][4] = mat[0,0]*(det_123_123) \
                 + mat[0,1]*(-det_123_023) \
                 + mat[0,2]*(det_123_013) \
                 + mat[0,3]*(-det_123_012)


        inv_det = 1.0/(mat[0,0]*tmp_inv[0][0] + mat[1,0]*tmp_inv[0][1] + mat[2,0]*tmp_inv[0][2] + mat[3,0]*tmp_inv[0][3] + mat[4,0]*tmp_inv[0][4])

        for i in range(k):
            for j in range(k):
                out_inv[i,j] = tmp_inv[i][j] * inv_det
    else:
        inverse_dgetr(mat, out_inv)

cdef void inverse_symmetric_psd(double[:,:] mat, double[:,:] out_inv):
    """
    Computes the inverse of a symmetric matrix `mat` in a matrix `out_inv`.
    
    **IMPORTANT NOTE: mat should be different to out_inv**.
    """
    cdef Py_ssize_t k = mat.shape[0], i, j
    cdef double inv_det
    cdef double tmp_inv[4]
    cdef double det_23_01,det_23_02,det_23_03,det_23_04,det_23_12,det_23_13,det_23_14,det_23_23,det_23_24,det_23_34,\
        det_24_01,det_24_02,det_24_03,det_24_04,det_24_12,det_24_13,det_24_14,det_24_23,det_24_24,det_24_34,det_34_01,\
        det_34_02,det_34_03,det_34_04,det_34_12,det_34_13,det_34_14,det_34_23,det_34_24,det_34_34,det_123_012,\
        det_123_013,det_123_014,det_123_023,det_123_024,det_123_034,det_123_123,det_123_124,det_123_134,det_123_234,\
        det_124_012,det_124_013,det_124_014,det_124_023,det_124_024,det_124_034,det_124_123,det_124_124,det_124_134,\
        det_124_234,det_134_013,det_134_014,det_134_023,det_134_024,det_134_034,det_134_123,det_134_124,det_134_134,\
        det_134_234,det_234_023,det_234_024,det_234_034,det_234_123,det_234_124,det_234_134,det_234_234

    if k == 2:
        inv_det = 1.0 / (mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0])

        tmp_inv[0] = mat[0,0]

        out_inv[0,0] = mat[1,1] * inv_det
        out_inv[0,1] = out_inv[1,0] = -mat[0,1] * inv_det
        out_inv[1,1] = tmp_inv[0] * inv_det

    elif k == 3:
        inv_det = 1.0 / (mat[0,0]*(mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) +
                         mat[0,1]*(mat[1,2]*mat[2,0] - mat[1,0]*mat[2,2]) +
                         mat[0,2]*(mat[1,0]*mat[2,1] - mat[1,1]*mat[2,0]) )

        tmp_inv[0] = mat[0,0]
        tmp_inv[1] = mat[1,1]

        out_inv[0,0] = (mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1])
        out_inv[0,1] = -(mat[0,1]*mat[2,2] - mat[0,2]*mat[2,1])
        out_inv[0,2] = (mat[1,0]*mat[1,2] - mat[0,2]*mat[1,1])
        out_inv[1,1] = (tmp_inv[0]*mat[2,2] - mat[2,0]*mat[2,0])
        out_inv[1,2] = -(tmp_inv[0]*mat[1,2] - mat[2,0]*mat[1,0])
        out_inv[2,2] = (tmp_inv[0]*tmp_inv[1] - mat[1,0]*mat[1,0])

        for i in range(k):
            for j in range(i, k):
                out_inv[i,j] = out_inv[i,j] * inv_det

        out_inv[1,0] = out_inv[0,1]
        out_inv[2,0] = out_inv[0,2]
        out_inv[2,1] = out_inv[1,2]

    elif k == 4:
        tmp_inv[0] = mat[0,0]
        tmp_inv[1] = mat[1,1]
        tmp_inv[2] = mat[2,2]

        out_inv[0,0] = (mat[1,1]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[1,2]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3]) +
                        mat[1,3]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]) )

        out_inv[0,1] = -(mat[0,1]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[0,2]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3]) +
                        mat[0,3]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]) )

        out_inv[0,2] = (mat[1,0]*(mat[1,2]*mat[3,3] - mat[1,3]*mat[3,2]) +
                        mat[0,2]*(mat[1,3]*mat[3,1] - mat[1,1]*mat[3,3]) +
                        mat[0,3]*(mat[1,1]*mat[3,2] - mat[1,2]*mat[3,1]) )

        out_inv[0,3] = -(mat[1,0]*(mat[1,2]*mat[2,3] - mat[1,3]*mat[2,2]) +
                        mat[2,0]*(mat[1,3]*mat[2,1] - mat[1,1]*mat[2,3]) +
                        mat[0,3]*(mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) )

        out_inv[1,1] = (tmp_inv[0]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[2,0]*(mat[2,3]*mat[3,0] - mat[2,0]*mat[3,3]) +
                        mat[3,0]*(mat[2,0]*mat[3,2] - mat[2,2]*mat[3,0]) )

        out_inv[1,2] = -(tmp_inv[0]*(mat[1,2]*mat[3,3] - mat[1,3]*mat[3,2]) +
                        mat[2,0]*(mat[1,3]*mat[3,0] - mat[1,0]*mat[3,3]) +
                        mat[3,0]*(mat[1,0]*mat[3,2] - mat[1,2]*mat[3,0]) )

        out_inv[1,3] = (tmp_inv[0]*(mat[2,1]*mat[2,3] - mat[1,3]*mat[2,2]) +
                        mat[2,0]*(mat[1,3]*mat[2,0] - mat[1,0]*mat[2,3]) +
                        mat[3,0]*(mat[1,0]*mat[2,2] - mat[2,1]*mat[2,0]) )

        out_inv[2,2] = (tmp_inv[0]*(tmp_inv[1]*mat[3,3] - mat[3,1]*mat[3,1]) +
                        mat[1,0]*(mat[3,1]*mat[3,0] - mat[1,0]*mat[3,3]) +
                        mat[3,0]*(mat[1,0]*mat[3,1] - tmp_inv[1]*mat[3,0]) )

        out_inv[2,3] = -(tmp_inv[0]*(tmp_inv[1]*mat[2,3] - mat[3,1]*mat[2,1]) +
                         mat[1,0]*(mat[3,1]*mat[2,0] - mat[1,0]*mat[2,3]) +
                         mat[3,0]*(mat[1,0]*mat[2,1] - tmp_inv[1]*mat[2,0]) )

        out_inv[3,3] = (tmp_inv[0]*(tmp_inv[1]*tmp_inv[2] - mat[2,1]*mat[2,1]) +
                        mat[1,0]*(mat[2,1]*mat[2,0] - mat[1,0]*tmp_inv[2]) +
                        mat[2,0]*(mat[1,0]*mat[2,1] - tmp_inv[1]*mat[2,0]) )

        inv_det = 1.0/(tmp_inv[0]*out_inv[0,0] + mat[1,0]*out_inv[0,1] + mat[2,0]*out_inv[0,2] + mat[3,0]*out_inv[0,3])

        for i in range(k):
            for j in range(i, k):
                out_inv[i,j] = out_inv[i,j] * inv_det

        out_inv[1,0] = out_inv[0,1]
        out_inv[2,0] = out_inv[0,2]
        out_inv[3,0] = out_inv[0,3]
        out_inv[2,1] = out_inv[1,2]
        out_inv[3,1] = out_inv[1,3]
        out_inv[3,2] = out_inv[2,3]

    elif k==5:
        tmp_inv[0] = mat[0,0]
        tmp_inv[1] = mat[1,1]
        tmp_inv[2] = mat[2,2]
        tmp_inv[3] = mat[3,3]

        det_23_01 = mat[2,0]*mat[3,1] - mat[2,1]*mat[3,0]
        det_23_02 = mat[2,0]*mat[3,2] - mat[2,2]*mat[3,0]
        det_23_03 = mat[2,0]*mat[3,3] - mat[2,3]*mat[3,0]
        det_23_04 = mat[2,0]*mat[3,4] - mat[2,4]*mat[3,0]
        det_23_12 = mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]
        det_23_13 = mat[2,1]*mat[3,3] - mat[2,3]*mat[3,1]
        det_23_14 = mat[2,1]*mat[3,4] - mat[2,4]*mat[3,1]
        det_23_23 = mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]
        det_23_24 = mat[2,2]*mat[3,4] - mat[2,4]*mat[3,2]
        det_23_34 = mat[2,3]*mat[3,4] - mat[2,4]*mat[3,3]
        det_24_01 = mat[2,0]*mat[4,1] - mat[2,1]*mat[4,0]
        det_24_02 = mat[2,0]*mat[4,2] - mat[2,2]*mat[4,0]
        det_24_03 = mat[2,0]*mat[4,3] - mat[2,3]*mat[4,0]
        det_24_04 = mat[2,0]*mat[4,4] - mat[2,4]*mat[4,0]
        det_24_12 = mat[2,1]*mat[4,2] - mat[2,2]*mat[4,1]
        det_24_13 = mat[2,1]*mat[4,3] - mat[2,3]*mat[4,1]
        det_24_14 = mat[2,1]*mat[4,4] - mat[2,4]*mat[4,1]
        det_24_23 = mat[2,2]*mat[4,3] - mat[2,3]*mat[4,2]
        det_24_24 = mat[2,2]*mat[4,4] - mat[2,4]*mat[4,2]
        det_24_34 = mat[2,3]*mat[4,4] - mat[2,4]*mat[4,3]
        det_34_01 = mat[3,0]*mat[4,1] - mat[3,1]*mat[4,0]
        det_34_02 = mat[3,0]*mat[4,2] - mat[3,2]*mat[4,0]
        det_34_03 = mat[3,0]*mat[4,3] - mat[3,3]*mat[4,0]
        det_34_04 = mat[3,0]*mat[4,4] - mat[3,4]*mat[4,0]
        det_34_12 = mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1]
        det_34_13 = mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1]
        det_34_14 = mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]
        det_34_23 = mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2]
        det_34_24 = mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]
        det_34_34 = mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]
        det_123_012 = mat[1,0]*(det_23_12) \
                 +mat[1,1]*(-det_23_02) \
                 +mat[1,2]*(det_23_01)
        det_123_013 = mat[1,0]*(det_23_13) \
                 +mat[1,1]*(-det_23_03) \
                 +mat[1,3]*(det_23_01)
        det_123_014 = mat[1,0]*(det_23_14) \
                 +mat[1,1]*(-det_23_04) \
                 +mat[1,4]*(det_23_01)
        det_123_023 = mat[1,0]*(det_23_23) \
                 +mat[1,2]*(-det_23_03) \
                 +mat[1,3]*(det_23_02)
        det_123_024 = mat[1,0]*(det_23_24) \
                 +mat[1,2]*(-det_23_04) \
                 +mat[1,4]*(det_23_02)
        det_123_034 = mat[1,0]*(det_23_34) \
                 +mat[1,3]*(-det_23_04) \
                 +mat[1,4]*(det_23_03)
        det_123_123 = mat[1,1]*(det_23_23) \
                 +mat[1,2]*(-det_23_13) \
                 +mat[1,3]*(det_23_12)
        det_123_124 = mat[1,1]*(det_23_24) \
                 +mat[1,2]*(-det_23_14) \
                 +mat[1,4]*(det_23_12)
        det_123_134 = mat[1,1]*(det_23_34) \
                 +mat[1,3]*(-det_23_14) \
                 +mat[1,4]*(det_23_13)
        det_123_234 = mat[1,2]*(det_23_34) \
                 +mat[1,3]*(-det_23_24) \
                 +mat[1,4]*(det_23_23)
        det_124_012 = mat[1,0]*(det_24_12) \
                 +mat[1,1]*(-det_24_02) \
                 +mat[1,2]*(det_24_01)
        det_124_013 = mat[1,0]*(det_24_13) \
                 +mat[1,1]*(-det_24_03) \
                 +mat[1,3]*(det_24_01)
        det_124_014 = mat[1,0]*(det_24_14) \
                 +mat[1,1]*(-det_24_04) \
                 +mat[1,4]*(det_24_01)
        det_124_023 = mat[1,0]*(det_24_23) \
                 +mat[1,2]*(-det_24_03) \
                 +mat[1,3]*(det_24_02)
        det_124_024 = mat[1,0]*(det_24_24) \
                 +mat[1,2]*(-det_24_04) \
                 +mat[1,4]*(det_24_02)
        det_124_034 = mat[1,0]*(det_24_34) \
                 +mat[1,3]*(-det_24_04) \
                 +mat[1,4]*(det_24_03)
        det_124_123 = mat[1,1]*(det_24_23) \
                 +mat[1,2]*(-det_24_13) \
                 +mat[1,3]*(det_24_12)
        det_124_124 = mat[1,1]*(det_24_24) \
                 +mat[1,2]*(-det_24_14) \
                 +mat[1,4]*(det_24_12)
        det_124_134 = mat[1,1]*(det_24_34) \
                 +mat[1,3]*(-det_24_14) \
                 +mat[1,4]*(det_24_13)
        det_124_234 = mat[1,2]*(det_24_34) \
                 +mat[1,3]*(-det_24_24) \
                 +mat[1,4]*(det_24_23)
        det_134_013 = mat[1,0]*(det_34_13) \
                 +mat[1,1]*(-det_34_03) \
                 +mat[1,3]*(det_34_01)
        det_134_014 = mat[1,0]*(det_34_14) \
                 +mat[1,1]*(-det_34_04) \
                 +mat[1,4]*(det_34_01)
        det_134_023 = mat[1,0]*(det_34_23) \
                 +mat[1,2]*(-det_34_03) \
                 +mat[1,3]*(det_34_02)
        det_134_024 = mat[1,0]*(det_34_24) \
                 +mat[1,2]*(-det_34_04) \
                 +mat[1,4]*(det_34_02)
        det_134_034 = mat[1,0]*(det_34_34) \
                 +mat[1,3]*(-det_34_04) \
                 +mat[1,4]*(det_34_03)
        det_134_123 = mat[1,1]*(det_34_23) \
                 +mat[1,2]*(-det_34_13) \
                 +mat[1,3]*(det_34_12)
        det_134_124 = mat[1,1]*(det_34_24) \
                 +mat[1,2]*(-det_34_14) \
                 +mat[1,4]*(det_34_12)
        det_134_134 = mat[1,1]*(det_34_34) \
                 +mat[1,3]*(-det_34_14) \
                 +mat[1,4]*(det_34_13)
        det_134_234 = mat[1,2]*(det_34_34) \
                 +mat[1,3]*(-det_34_24) \
                 +mat[1,4]*(det_34_23)
        det_234_023 = mat[2,0]*(det_34_23) \
                 +mat[2,2]*(-det_34_03) \
                 +mat[2,3]*(det_34_02)
        det_234_024 = mat[2,0]*(det_34_24) \
                 +mat[2,2]*(-det_34_04) \
                 +mat[2,4]*(det_34_02)
        det_234_034 = mat[2,0]*(det_34_34) \
                 +mat[2,3]*(-det_34_04) \
                 +mat[2,4]*(det_34_03)
        det_234_123 = mat[2,1]*(det_34_23) \
                 +mat[2,2]*(-det_34_13) \
                 +mat[2,3]*(det_34_12)
        det_234_124 = mat[2,1]*(det_34_24) \
                 +mat[2,2]*(-det_34_14) \
                 +mat[2,4]*(det_34_12)
        det_234_134 = mat[2,1]*(det_34_34) \
                 +mat[2,3]*(-det_34_14) \
                 +mat[2,4]*(det_34_13)
        det_234_234 = mat[2,2]*(det_34_34) \
                 +mat[2,3]*(-det_34_24) \
                 +mat[2,4]*(det_34_23)

        out_inv[0,0] = mat[1,1]*(det_234_234) \
                 + mat[2,1]*(-det_234_134) \
                 + mat[3,1]*(det_234_124) \
                 + mat[4,1]*(-det_234_123)

        out_inv[0,1] = mat[1,0]*(-det_234_234) \
                 + mat[2,0]*(det_234_134) \
                 + mat[3,0]*(-det_234_124) \
                 + mat[4,0]*(det_234_123)

        out_inv[0,2] = mat[1,0]*(det_134_234) \
                 + mat[2,0]*(-det_134_134) \
                 + mat[3,0]*(det_134_124) \
                 + mat[4,0]*(-det_134_123)

        out_inv[0,3] = mat[1,0]*(-det_124_234) \
                 + mat[2,0]*(det_124_134) \
                 + mat[3,0]*(-det_124_124) \
                 + mat[4,0]*(det_124_123)

        out_inv[0,4] = mat[1,0]*(det_123_234) \
                 + mat[2,0]*(-det_123_134) \
                 + mat[3,0]*(det_123_124) \
                 + mat[4,0]*(-det_123_123)

        out_inv[1,1] = tmp_inv[0]*(det_234_234) \
                 + mat[2,0]*(-det_234_034) \
                 + mat[3,0]*(det_234_024) \
                 + mat[4,0]*(-det_234_023)

        out_inv[1,2] = tmp_inv[0]*(-det_134_234) \
                 + mat[2,0]*(det_134_034) \
                 + mat[3,0]*(-det_134_024) \
                 + mat[4,0]*(det_134_023)

        out_inv[1,3] = tmp_inv[0]*(det_124_234) \
                 + mat[2,0]*(-det_124_034) \
                 + mat[3,0]*(det_124_024) \
                 + mat[4,0]*(-det_124_023)

        out_inv[1,4] = tmp_inv[0]*(-det_123_234) \
                 + mat[2,0]*(det_123_034) \
                 + mat[3,0]*(-det_123_024) \
                 + mat[4,0]*(det_123_023)

        out_inv[2,2] = tmp_inv[0]*(det_134_134) \
                 + mat[1,0]*(-det_134_034) \
                 + mat[3,0]*(det_134_014) \
                 + mat[4,0]*(-det_134_013)

        out_inv[2,3] = tmp_inv[0]*(-det_124_134) \
                 + mat[1,0]*(det_124_034) \
                 + mat[3,0]*(-det_124_014) \
                 + mat[4,0]*(det_124_013)

        out_inv[2,4] = tmp_inv[0]*(det_123_134) \
                 + mat[1,0]*(-det_123_034) \
                 + mat[3,0]*(det_123_014) \
                 + mat[4,0]*(-det_123_013)

        out_inv[3,3] = tmp_inv[0]*(det_124_124) \
                 + mat[1,0]*(-det_124_024) \
                 + mat[2,0]*(det_124_014) \
                 + mat[4,0]*(-det_124_012)

        out_inv[3,4] = tmp_inv[0]*(-det_123_124) \
                 + mat[1,0]*(det_123_024) \
                 + mat[2,0]*(-det_123_014) \
                 + mat[4,0]*(det_123_012)

        out_inv[4,4] = tmp_inv[0]*(det_123_123) \
                 + mat[1,0]*(-det_123_023) \
                 + mat[2,0]*(det_123_013) \
                 + mat[3,0]*(-det_123_012)

        inv_det = 1.0/(tmp_inv[0]*out_inv[0,0] + mat[1,0]*out_inv[0,1] + mat[2,0]*out_inv[0,2] + mat[3,0]*out_inv[0,3] + mat[4,0]*out_inv[0,4])

        for i in range(k):
            for j in range(i,k):
                out_inv[i,j] = out_inv[i,j] * inv_det

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

cdef void dgemv(double alpha, double[:,:] A, double[:] x, double beta, double[:] y):
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

cdef double lapack_det(double[:,:] mat, bint overwrite):
    cdef int k = mat.shape[0]
    cdef int* ipiv = <int*>malloc(sizeof(int)*k)
    cdef int info = 0
    cdef Py_ssize_t i
    cdef double det = 1
    cdef int sign = 1

    cdef double* mat_ptr = &mat[0,0]
    if not overwrite:
        mat_ptr = <double*> malloc(sizeof(double)*mat.shape[0]*mat.shape[1])
        memcpy(mat_ptr, &mat[0,0], sizeof(double)*mat.shape[0]*mat.shape[1])

    try:
        cython_lapack.dgetrf(&k, &k, mat_ptr, &k, ipiv, &info)
        if info > 0:
            return 0
        elif info == 0:
            for i in range(k):
                if (ipiv[i] != (i + 1)):
                    sign = -sign
                det *= mat_ptr[i*k+i]

            return sign*det
        else:
            sys.exit("Unsuccesful determinant.")
    finally:
        free(ipiv)
        if not overwrite:
            free(mat_ptr)

cdef double det(double[:,:] mat, bint overwrite=False):
    cdef double det_23_12,det_23_13,det_23_14,det_23_23,det_23_24,det_23_34,det_24_12,det_24_13,det_24_14,det_24_23,\
        det_24_24,det_24_34,det_34_12,det_34_13,det_34_14,det_34_23,det_34_24,det_34_34,det_123_123,det_123_124,\
        det_123_134,det_123_234,det_124_123,det_124_124,det_124_134,det_124_234,det_134_123,det_134_124,det_134_134,\
        det_134_234,det_234_123,det_234_124,det_234_134,det_234_234

    cdef double det_34_15,det_34_25,det_34_35,det_34_45,\
        det_35_12,det_35_13,det_35_14,det_35_15,det_35_23,det_35_24,det_35_25,det_35_34,det_35_35,det_35_45,det_45_12,\
        det_45_13,det_45_14,det_45_15,det_45_23,det_45_24,det_45_25,det_45_34,det_45_35,det_45_45,\
        det_234_125,det_234_135,det_234_145,det_234_235,det_234_245,det_234_345,\
        det_235_123,det_235_124,det_235_125,det_235_134,det_235_135,det_235_145,det_235_234,det_235_235,det_235_245,\
        det_235_345,det_245_123,det_245_124,det_245_125,det_245_134,det_245_135,det_245_145,det_245_234,det_245_235,\
        det_245_245,det_245_345,det_345_123,det_345_124,det_345_125,det_345_134,det_345_135,det_345_145,det_345_234,\
        det_345_235,det_345_245,det_345_345,det_1234_1234,det_1234_1235,det_1234_1245,det_1234_1345,det_1234_2345,\
        det_1235_1234,det_1235_1235,det_1235_1245,det_1235_1345,det_1235_2345,det_1245_1234,det_1245_1235,det_1245_1245,\
        det_1245_1345,det_1245_2345,det_1345_1234,det_1345_1235,det_1345_1245,det_1345_1345,det_1345_2345,det_2345_1234,\
        det_2345_1235,det_2345_1245,det_2345_1345,det_2345_2345

    if mat.shape[0] == 1:
        return mat[0,0]
    elif mat.shape[0] == 2:
        return mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0]
    elif mat.shape[0] == 3:
        return mat[0,0]*mat[1,1]*mat[2,2] + mat[0,1]*mat[1,2]*mat[2,0] + mat[0,2]*mat[1,0]*mat[2,1] \
               - mat[0,2]*mat[1,1]*mat[2,0] - mat[0,1]*mat[1,0]*mat[2,2] - mat[0,0]*mat[1,2]*mat[2,1]
    elif mat.shape[0] == 4:
        return mat[0,0]*(mat[1,1]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                        mat[1,2]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3]) +
                        mat[1,3]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1])) \
                + mat[1,0]*-(mat[0,1]*(mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]) +
                                mat[0,2]*(mat[2,3]*mat[3,1] - mat[2,1]*mat[3,3]) +
                                mat[0,3]*(mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]) ) \
                + mat[2,0]*(mat[0,1]*(mat[1,2]*mat[3,3] - mat[1,3]*mat[3,2]) +
                                mat[0,2]*(mat[1,3]*mat[3,1] - mat[1,1]*mat[3,3]) +
                                mat[0,3]*(mat[1,1]*mat[3,2] - mat[1,2]*mat[3,1]) ) \
                + mat[3,0]*-(mat[0,1]*(mat[1,2]*mat[2,3] - mat[1,3]*mat[2,2]) +
                                mat[0,2]*(mat[1,3]*mat[2,1] - mat[1,1]*mat[2,3]) +
                                mat[0,3]*(mat[1,1]*mat[2,2] - mat[1,2]*mat[2,1]) )
    elif mat.shape[0] == 5:
        det_23_12 = mat[2,1]*mat[3,2] - mat[2,2]*mat[3,1]
        det_23_13 = mat[2,1]*mat[3,3] - mat[2,3]*mat[3,1]
        det_23_14 = mat[2,1]*mat[3,4] - mat[2,4]*mat[3,1]
        det_23_23 = mat[2,2]*mat[3,3] - mat[2,3]*mat[3,2]
        det_23_24 = mat[2,2]*mat[3,4] - mat[2,4]*mat[3,2]
        det_23_34 = mat[2,3]*mat[3,4] - mat[2,4]*mat[3,3]
        det_24_12 = mat[2,1]*mat[4,2] - mat[2,2]*mat[4,1]
        det_24_13 = mat[2,1]*mat[4,3] - mat[2,3]*mat[4,1]
        det_24_14 = mat[2,1]*mat[4,4] - mat[2,4]*mat[4,1]
        det_24_23 = mat[2,2]*mat[4,3] - mat[2,3]*mat[4,2]
        det_24_24 = mat[2,2]*mat[4,4] - mat[2,4]*mat[4,2]
        det_24_34 = mat[2,3]*mat[4,4] - mat[2,4]*mat[4,3]
        det_34_12 = mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1]
        det_34_13 = mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1]
        det_34_14 = mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]
        det_34_23 = mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2]
        det_34_24 = mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]
        det_34_34 = mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]
        det_123_123 = mat[1,1]*(det_23_23) \
                 +mat[1,2]*(-det_23_13) \
                 +mat[1,3]*(det_23_12)
        det_123_124 = mat[1,1]*(det_23_24) \
                 +mat[1,2]*(-det_23_14) \
                 +mat[1,4]*(det_23_12)
        det_123_134 = mat[1,1]*(det_23_34) \
                 +mat[1,3]*(-det_23_14) \
                 +mat[1,4]*(det_23_13)
        det_123_234 = mat[1,2]*(det_23_34) \
                 +mat[1,3]*(-det_23_24) \
                 +mat[1,4]*(det_23_23)
        det_124_123 = mat[1,1]*(det_24_23) \
                 +mat[1,2]*(-det_24_13) \
                 +mat[1,3]*(det_24_12)
        det_124_124 = mat[1,1]*(det_24_24) \
                 +mat[1,2]*(-det_24_14) \
                 +mat[1,4]*(det_24_12)
        det_124_134 = mat[1,1]*(det_24_34) \
                 +mat[1,3]*(-det_24_14) \
                 +mat[1,4]*(det_24_13)
        det_124_234 = mat[1,2]*(det_24_34) \
                 +mat[1,3]*(-det_24_24) \
                 +mat[1,4]*(det_24_23)
        det_134_123 = mat[1,1]*(det_34_23) \
                 +mat[1,2]*(-det_34_13) \
                 +mat[1,3]*(det_34_12)
        det_134_124 = mat[1,1]*(det_34_24) \
                 +mat[1,2]*(-det_34_14) \
                 +mat[1,4]*(det_34_12)
        det_134_134 = mat[1,1]*(det_34_34) \
                 +mat[1,3]*(-det_34_14) \
                 +mat[1,4]*(det_34_13)
        det_134_234 = mat[1,2]*(det_34_34) \
                 +mat[1,3]*(-det_34_24) \
                 +mat[1,4]*(det_34_23)
        det_234_123 = mat[2,1]*(det_34_23) \
                 +mat[2,2]*(-det_34_13) \
                 +mat[2,3]*(det_34_12)
        det_234_124 = mat[2,1]*(det_34_24) \
                 +mat[2,2]*(-det_34_14) \
                 +mat[2,4]*(det_34_12)
        det_234_134 = mat[2,1]*(det_34_34) \
                 +mat[2,3]*(-det_34_14) \
                 +mat[2,4]*(det_34_13)
        det_234_234 = mat[2,2]*(det_34_34) \
                 +mat[2,3]*(-det_34_24) \
                 +mat[2,4]*(det_34_23)

        return mat[0,0]*(mat[1,1]*(det_234_234) \
                         + mat[1,2]*(-det_234_134) \
                         + mat[1,3]*(det_234_124) \
                         + mat[1,4]*(-det_234_123)) \
               + mat[1,0]*(mat[0,1]*(-det_234_234) \
                         + mat[0,2]*(det_234_134) \
                         + mat[0,3]*(-det_234_124) \
                         + mat[0,4]*(det_234_123)) \
               + mat[2,0]*(mat[0,1]*(det_134_234) \
                         + mat[0,2]*(-det_134_134) \
                         + mat[0,3]*(det_134_124) \
                         + mat[0,4]*(-det_134_123)) \
               + mat[3,0]*(mat[0,1]*(-det_124_234) \
                         + mat[0,2]*(det_124_134) \
                         + mat[0,3]*(-det_124_124) \
                         + mat[0,4]*(det_124_123)) \
               + mat[4,0]*(mat[0,1]*(det_123_234) \
                         + mat[0,2]*(-det_123_134) \
                         + mat[0,3]*(det_123_124) \
                         + mat[0,4]*(-det_123_123))
    elif mat.shape[0] == 6:
        det_34_12 = mat[3,1]*mat[4,2] - mat[3,2]*mat[4,1]
        det_34_13 = mat[3,1]*mat[4,3] - mat[3,3]*mat[4,1]
        det_34_14 = mat[3,1]*mat[4,4] - mat[3,4]*mat[4,1]
        det_34_15 = mat[3,1]*mat[4,5] - mat[3,5]*mat[4,1]
        det_34_23 = mat[3,2]*mat[4,3] - mat[3,3]*mat[4,2]
        det_34_24 = mat[3,2]*mat[4,4] - mat[3,4]*mat[4,2]
        det_34_25 = mat[3,2]*mat[4,5] - mat[3,5]*mat[4,2]
        det_34_34 = mat[3,3]*mat[4,4] - mat[3,4]*mat[4,3]
        det_34_35 = mat[3,3]*mat[4,5] - mat[3,5]*mat[4,3]
        det_34_45 = mat[3,4]*mat[4,5] - mat[3,5]*mat[4,4]
        det_35_12 = mat[3,1]*mat[5,2] - mat[3,2]*mat[5,1]
        det_35_13 = mat[3,1]*mat[5,3] - mat[3,3]*mat[5,1]
        det_35_14 = mat[3,1]*mat[5,4] - mat[3,4]*mat[5,1]
        det_35_15 = mat[3,1]*mat[5,5] - mat[3,5]*mat[5,1]
        det_35_23 = mat[3,2]*mat[5,3] - mat[3,3]*mat[5,2]
        det_35_24 = mat[3,2]*mat[5,4] - mat[3,4]*mat[5,2]
        det_35_25 = mat[3,2]*mat[5,5] - mat[3,5]*mat[5,2]
        det_35_34 = mat[3,3]*mat[5,4] - mat[3,4]*mat[5,3]
        det_35_35 = mat[3,3]*mat[5,5] - mat[3,5]*mat[5,3]
        det_35_45 = mat[3,4]*mat[5,5] - mat[3,5]*mat[5,4]
        det_45_12 = mat[4,1]*mat[5,2] - mat[4,2]*mat[5,1]
        det_45_13 = mat[4,1]*mat[5,3] - mat[4,3]*mat[5,1]
        det_45_14 = mat[4,1]*mat[5,4] - mat[4,4]*mat[5,1]
        det_45_15 = mat[4,1]*mat[5,5] - mat[4,5]*mat[5,1]
        det_45_23 = mat[4,2]*mat[5,3] - mat[4,3]*mat[5,2]
        det_45_24 = mat[4,2]*mat[5,4] - mat[4,4]*mat[5,2]
        det_45_25 = mat[4,2]*mat[5,5] - mat[4,5]*mat[5,2]
        det_45_34 = mat[4,3]*mat[5,4] - mat[4,4]*mat[5,3]
        det_45_35 = mat[4,3]*mat[5,5] - mat[4,5]*mat[5,3]
        det_45_45 = mat[4,4]*mat[5,5] - mat[4,5]*mat[5,4]
        det_234_123 = mat[2,1]*(det_34_23) \
                 +mat[2,2]*(-det_34_13) \
                 +mat[2,3]*(det_34_12)
        det_234_124 = mat[2,1]*(det_34_24) \
                 +mat[2,2]*(-det_34_14) \
                 +mat[2,4]*(det_34_12)
        det_234_125 = mat[2,1]*(det_34_25) \
                 +mat[2,2]*(-det_34_15) \
                 +mat[2,5]*(det_34_12)
        det_234_134 = mat[2,1]*(det_34_34) \
                 +mat[2,3]*(-det_34_14) \
                 +mat[2,4]*(det_34_13)
        det_234_135 = mat[2,1]*(det_34_35) \
                 +mat[2,3]*(-det_34_15) \
                 +mat[2,5]*(det_34_13)
        det_234_145 = mat[2,1]*(det_34_45) \
                 +mat[2,4]*(-det_34_15) \
                 +mat[2,5]*(det_34_14)
        det_234_234 = mat[2,2]*(det_34_34) \
                 +mat[2,3]*(-det_34_24) \
                 +mat[2,4]*(det_34_23)
        det_234_235 = mat[2,2]*(det_34_35) \
                 +mat[2,3]*(-det_34_25) \
                 +mat[2,5]*(det_34_23)
        det_234_245 = mat[2,2]*(det_34_45) \
                 +mat[2,4]*(-det_34_25) \
                 +mat[2,5]*(det_34_24)
        det_234_345 = mat[2,3]*(det_34_45) \
                 +mat[2,4]*(-det_34_35) \
                 +mat[2,5]*(det_34_34)
        det_235_123 = mat[2,1]*(det_35_23) \
                 +mat[2,2]*(-det_35_13) \
                 +mat[2,3]*(det_35_12)
        det_235_124 = mat[2,1]*(det_35_24) \
                 +mat[2,2]*(-det_35_14) \
                 +mat[2,4]*(det_35_12)
        det_235_125 = mat[2,1]*(det_35_25) \
                 +mat[2,2]*(-det_35_15) \
                 +mat[2,5]*(det_35_12)
        det_235_134 = mat[2,1]*(det_35_34) \
                 +mat[2,3]*(-det_35_14) \
                 +mat[2,4]*(det_35_13)
        det_235_135 = mat[2,1]*(det_35_35) \
                 +mat[2,3]*(-det_35_15) \
                 +mat[2,5]*(det_35_13)
        det_235_145 = mat[2,1]*(det_35_45) \
                 +mat[2,4]*(-det_35_15) \
                 +mat[2,5]*(det_35_14)
        det_235_234 = mat[2,2]*(det_35_34) \
                 +mat[2,3]*(-det_35_24) \
                 +mat[2,4]*(det_35_23)
        det_235_235 = mat[2,2]*(det_35_35) \
                 +mat[2,3]*(-det_35_25) \
                 +mat[2,5]*(det_35_23)
        det_235_245 = mat[2,2]*(det_35_45) \
                 +mat[2,4]*(-det_35_25) \
                 +mat[2,5]*(det_35_24)
        det_235_345 = mat[2,3]*(det_35_45) \
                 +mat[2,4]*(-det_35_35) \
                 +mat[2,5]*(det_35_34)
        det_245_123 = mat[2,1]*(det_45_23) \
                 +mat[2,2]*(-det_45_13) \
                 +mat[2,3]*(det_45_12)
        det_245_124 = mat[2,1]*(det_45_24) \
                 +mat[2,2]*(-det_45_14) \
                 +mat[2,4]*(det_45_12)
        det_245_125 = mat[2,1]*(det_45_25) \
                 +mat[2,2]*(-det_45_15) \
                 +mat[2,5]*(det_45_12)
        det_245_134 = mat[2,1]*(det_45_34) \
                 +mat[2,3]*(-det_45_14) \
                 +mat[2,4]*(det_45_13)
        det_245_135 = mat[2,1]*(det_45_35) \
                 +mat[2,3]*(-det_45_15) \
                 +mat[2,5]*(det_45_13)
        det_245_145 = mat[2,1]*(det_45_45) \
                 +mat[2,4]*(-det_45_15) \
                 +mat[2,5]*(det_45_14)
        det_245_234 = mat[2,2]*(det_45_34) \
                 +mat[2,3]*(-det_45_24) \
                 +mat[2,4]*(det_45_23)
        det_245_235 = mat[2,2]*(det_45_35) \
                 +mat[2,3]*(-det_45_25) \
                 +mat[2,5]*(det_45_23)
        det_245_245 = mat[2,2]*(det_45_45) \
                 +mat[2,4]*(-det_45_25) \
                 +mat[2,5]*(det_45_24)
        det_245_345 = mat[2,3]*(det_45_45) \
                 +mat[2,4]*(-det_45_35) \
                 +mat[2,5]*(det_45_34)
        det_345_123 = mat[3,1]*(det_45_23) \
                 +mat[3,2]*(-det_45_13) \
                 +mat[3,3]*(det_45_12)
        det_345_124 = mat[3,1]*(det_45_24) \
                 +mat[3,2]*(-det_45_14) \
                 +mat[3,4]*(det_45_12)
        det_345_125 = mat[3,1]*(det_45_25) \
                 +mat[3,2]*(-det_45_15) \
                 +mat[3,5]*(det_45_12)
        det_345_134 = mat[3,1]*(det_45_34) \
                 +mat[3,3]*(-det_45_14) \
                 +mat[3,4]*(det_45_13)
        det_345_135 = mat[3,1]*(det_45_35) \
                 +mat[3,3]*(-det_45_15) \
                 +mat[3,5]*(det_45_13)
        det_345_145 = mat[3,1]*(det_45_45) \
                 +mat[3,4]*(-det_45_15) \
                 +mat[3,5]*(det_45_14)
        det_345_234 = mat[3,2]*(det_45_34) \
                 +mat[3,3]*(-det_45_24) \
                 +mat[3,4]*(det_45_23)
        det_345_235 = mat[3,2]*(det_45_35) \
                 +mat[3,3]*(-det_45_25) \
                 +mat[3,5]*(det_45_23)
        det_345_245 = mat[3,2]*(det_45_45) \
                 +mat[3,4]*(-det_45_25) \
                 +mat[3,5]*(det_45_24)
        det_345_345 = mat[3,3]*(det_45_45) \
                 +mat[3,4]*(-det_45_35) \
                 +mat[3,5]*(det_45_34)
        det_1234_1234 = mat[1,1]*(det_234_234) \
                 +mat[1,2]*(-det_234_134) \
                 +mat[1,3]*(det_234_124) \
                 +mat[1,4]*(-det_234_123)
        det_1234_1235 = mat[1,1]*(det_234_235) \
                 +mat[1,2]*(-det_234_135) \
                 +mat[1,3]*(det_234_125) \
                 +mat[1,5]*(-det_234_123)
        det_1234_1245 = mat[1,1]*(det_234_245) \
                 +mat[1,2]*(-det_234_145) \
                 +mat[1,4]*(det_234_125) \
                 +mat[1,5]*(-det_234_124)
        det_1234_1345 = mat[1,1]*(det_234_345) \
                 +mat[1,3]*(-det_234_145) \
                 +mat[1,4]*(det_234_135) \
                 +mat[1,5]*(-det_234_134)
        det_1234_2345 = mat[1,2]*(det_234_345) \
                 +mat[1,3]*(-det_234_245) \
                 +mat[1,4]*(det_234_235) \
                 +mat[1,5]*(-det_234_234)
        det_1235_1234 = mat[1,1]*(det_235_234) \
                 +mat[1,2]*(-det_235_134) \
                 +mat[1,3]*(det_235_124) \
                 +mat[1,4]*(-det_235_123)
        det_1235_1235 = mat[1,1]*(det_235_235) \
                 +mat[1,2]*(-det_235_135) \
                 +mat[1,3]*(det_235_125) \
                 +mat[1,5]*(-det_235_123)
        det_1235_1245 = mat[1,1]*(det_235_245) \
                 +mat[1,2]*(-det_235_145) \
                 +mat[1,4]*(det_235_125) \
                 +mat[1,5]*(-det_235_124)
        det_1235_1345 = mat[1,1]*(det_235_345) \
                 +mat[1,3]*(-det_235_145) \
                 +mat[1,4]*(det_235_135) \
                 +mat[1,5]*(-det_235_134)
        det_1235_2345 = mat[1,2]*(det_235_345) \
                 +mat[1,3]*(-det_235_245) \
                 +mat[1,4]*(det_235_235) \
                 +mat[1,5]*(-det_235_234)
        det_1245_1234 = mat[1,1]*(det_245_234) \
                 +mat[1,2]*(-det_245_134) \
                 +mat[1,3]*(det_245_124) \
                 +mat[1,4]*(-det_245_123)
        det_1245_1235 = mat[1,1]*(det_245_235) \
                 +mat[1,2]*(-det_245_135) \
                 +mat[1,3]*(det_245_125) \
                 +mat[1,5]*(-det_245_123)
        det_1245_1245 = mat[1,1]*(det_245_245) \
                 +mat[1,2]*(-det_245_145) \
                 +mat[1,4]*(det_245_125) \
                 +mat[1,5]*(-det_245_124)
        det_1245_1345 = mat[1,1]*(det_245_345) \
                 +mat[1,3]*(-det_245_145) \
                 +mat[1,4]*(det_245_135) \
                 +mat[1,5]*(-det_245_134)
        det_1245_2345 = mat[1,2]*(det_245_345) \
                 +mat[1,3]*(-det_245_245) \
                 +mat[1,4]*(det_245_235) \
                 +mat[1,5]*(-det_245_234)
        det_1345_1234 = mat[1,1]*(det_345_234) \
                 +mat[1,2]*(-det_345_134) \
                 +mat[1,3]*(det_345_124) \
                 +mat[1,4]*(-det_345_123)
        det_1345_1235 = mat[1,1]*(det_345_235) \
                 +mat[1,2]*(-det_345_135) \
                 +mat[1,3]*(det_345_125) \
                 +mat[1,5]*(-det_345_123)
        det_1345_1245 = mat[1,1]*(det_345_245) \
                 +mat[1,2]*(-det_345_145) \
                 +mat[1,4]*(det_345_125) \
                 +mat[1,5]*(-det_345_124)
        det_1345_1345 = mat[1,1]*(det_345_345) \
                 +mat[1,3]*(-det_345_145) \
                 +mat[1,4]*(det_345_135) \
                 +mat[1,5]*(-det_345_134)
        det_1345_2345 = mat[1,2]*(det_345_345) \
                 +mat[1,3]*(-det_345_245) \
                 +mat[1,4]*(det_345_235) \
                 +mat[1,5]*(-det_345_234)
        det_2345_1234 = mat[2,1]*(det_345_234) \
                 +mat[2,2]*(-det_345_134) \
                 +mat[2,3]*(det_345_124) \
                 +mat[2,4]*(-det_345_123)
        det_2345_1235 = mat[2,1]*(det_345_235) \
                 +mat[2,2]*(-det_345_135) \
                 +mat[2,3]*(det_345_125) \
                 +mat[2,5]*(-det_345_123)
        det_2345_1245 = mat[2,1]*(det_345_245) \
                 +mat[2,2]*(-det_345_145) \
                 +mat[2,4]*(det_345_125) \
                 +mat[2,5]*(-det_345_124)
        det_2345_1345 = mat[2,1]*(det_345_345) \
                 +mat[2,3]*(-det_345_145) \
                 +mat[2,4]*(det_345_135) \
                 +mat[2,5]*(-det_345_134)
        det_2345_2345 = mat[2,2]*(det_345_345) \
                 +mat[2,3]*(-det_345_245) \
                 +mat[2,4]*(det_345_235) \
                 +mat[2,5]*(-det_345_234)

        return mat[0,0]*(mat[1,1]*(det_2345_2345) \
                         + mat[1,2]*(-det_2345_1345) \
                         + mat[1,3]*(det_2345_1245) \
                         + mat[1,4]*(-det_2345_1235) \
                         + mat[1,5]*(det_2345_1234)) \
             + mat[1,0]*(mat[0,1]*(-det_2345_2345) \
                         + mat[0,2]*(det_2345_1345) \
                         + mat[0,3]*(-det_2345_1245) \
                         + mat[0,4]*(det_2345_1235) \
                         + mat[0,5]*(-det_2345_1234)) \
             + mat[2,0]*(mat[0,1]*(det_1345_2345) \
                         + mat[0,2]*(-det_1345_1345) \
                         + mat[0,3]*(det_1345_1245) \
                         + mat[0,4]*(-det_1345_1235) \
                         + mat[0,5]*(det_1345_1234)) \
             + mat[3,0]*(mat[0,1]*(-det_1245_2345) \
                         + mat[0,2]*(det_1245_1345) \
                         + mat[0,3]*(-det_1245_1245) \
                         + mat[0,4]*(det_1245_1235) \
                         + mat[0,5]*(-det_1245_1234)) \
             + mat[4,0]*(mat[0,1]*(det_1235_2345) \
                         + mat[0,2]*(-det_1235_1345) \
                         + mat[0,3]*(det_1235_1245) \
                         + mat[0,4]*(-det_1235_1235) \
                         + mat[0,5]*(det_1235_1234)) \
             + mat[5,0]*(mat[0,1]*(-det_1234_2345) \
                         + mat[0,2]*(det_1234_1345) \
                         + mat[0,3]*(-det_1234_1245) \
                         + mat[0,4]*(det_1234_1235) \
                         + mat[0,5]*(-det_1234_1234))
    else:
        return lapack_det(mat, overwrite)