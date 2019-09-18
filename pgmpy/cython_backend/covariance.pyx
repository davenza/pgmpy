#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np

cdef double mean(double[:] data) nogil:
    cdef double mean = 0
    cdef Py_ssize_t N = data.shape[0], i

    for i in range(N):
        mean += data[i]

    return mean / N

cdef double[:] mean_vec(double[:,:] data):
    cdef Py_ssize_t N = data.shape[0]
    cdef Py_ssize_t k = data.shape[1]

    cdef Py_ssize_t i, j

    cdef double[:] means = np.zeros((k,))

    cdef double sum = 0
    for j in range(k):
        sum = 0
        for i in range(N):
            sum += data[i,j]

        means[j] = sum / N
    return means

cdef double sse(double[:] data, double mean):
    cdef double sse = 0, d
    cdef Py_ssize_t N = data.shape[0], i

    for i in range(N):
        d = data[i] - mean
        sse += d*d

    return sse

cdef double[:,:] sse_mat(double[:,:] data, double[:] means):
    cdef Py_ssize_t N = data.shape[0]
    cdef Py_ssize_t k = data.shape[1]
    cdef Py_ssize_t i, j, m

    cdef double tmp
    cdef double[:,:] cov = np.zeros((k,k))

    for i in range(k):
        for j in range(i,k):
            for m in range(N):
                tmp = (data[m,i] - means[i])
                cov[i,j] += tmp*tmp

            cov[i,j] = cov[j,i]
    return cov

cdef double[:,:] covariance(double[:,:] data, double[:] means):
    cdef Py_ssize_t N = data.shape[0]
    cdef Py_ssize_t k = data.shape[1]
    cdef Py_ssize_t i, j, m

    cdef double tmp
    cdef double[:,:] cov = np.zeros((k,k))

    for i in range(k):
        for j in range(i,k):
            for m in range(N):
                tmp = (data[m,i] - means[i])
                cov[i,j] += tmp*tmp

            cov[i,j] = cov[j,i] = cov[i,j] / (N-1)
    return cov

cdef double[:,:] covariance_mat_with_vec(double[:,:] data_mat, double[:] data_vec, double[:] means_mat, double mean_vec):
    cdef Py_ssize_t N = data_mat.shape[0]
    cdef Py_ssize_t k = data_mat.shape[1]
    cdef Py_ssize_t i, j, m

    cdef double tmp_mat, tmp_vec
    cdef double[:,:] cov = np.zeros((k+1,k+1))

    for i in range(0,k):
        for j in range(i,k):
            for m in range(N):
                cov[i+1,j+1] += (data_mat[m,i] - means_mat[i])*(data_mat[m,j] - means_mat[j])

    for i in range(0,k):
        for m in range(N):
            cov[0,i+1] += (data_mat[m,i] - means_mat[i])*(data_vec[m] - mean_vec)

    for m in range(N):
        cov[0,0] += (data_vec[m] - mean_vec)*(data_vec[m] - mean_vec)

    for i in range(k+1):
        for j in range(i+1,k+1):
            cov[j,i] = cov[i,j] = cov[i,j] / (N-1)

    return cov

cdef double[:,:] sse_mat_with_vec(double[:,:] data_mat, double[:] data_vec, double[:] means_mat, double mean_vec):
    cdef Py_ssize_t N = data_mat.shape[0]
    cdef Py_ssize_t k = data_mat.shape[1]
    cdef Py_ssize_t i, j, m

    cdef double tmp_mat, tmp_vec
    cdef double[:,:] sse = np.zeros((k+1,k+1))

    for i in range(0,k):
        for j in range(i,k):
            for m in range(N):
                sse[i+1,j+1] += (data_mat[m,i] - means_mat[i])*(data_mat[m,j] - means_mat[j])

    for i in range(0,k):
        for m in range(N):
            sse[0,i+1] += (data_mat[m,i] - means_mat[i])*(data_vec[m] - mean_vec)

    for m in range(N):
        sse[0,0] += (data_vec[m] - mean_vec)*(data_vec[m] - mean_vec)

    for i in range(k+1):
        for j in range(i+1,k+1):
            sse[j,i] = sse[i,j]

    return sse

cdef double[:,:] drop_variable(double[:,:] cov, int index_to_drop):
    cdef Py_ssize_t k = cov.shape[0], i, j

    cdef double[:,:] dropped = np.empty((k-1, k-1))

    for i in range(index_to_drop):
        for j in range(index_to_drop):
            dropped[i,j] = cov[i,j]

        for j in range(index_to_drop+1, k):
            dropped[i,j-1] = cov[i,j]

    for i in range(index_to_drop+1, k):
        for j in range(index_to_drop):
            dropped[i-1,j] = cov[i,j]

        for j in range(index_to_drop+1, k):
            dropped[i-1,j-1] = cov[i,j]

    return dropped