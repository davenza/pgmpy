#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
cimport covariance as covariance_fallback

import sys

DEF SIMD_LANES = 4
DEF SIMD_ALIGNMENT = 32

# TODO: Finish the implementation for every function.

cdef double mean(double[:] data) nogil:
    # Non-contiguous data. Fallback to scalar code.
    if data.strides[0] != sizeof(double):
        return covariance_fallback.mean(data)

    cdef Py_ssize_t N = data.shape[0], i, j = 0

    cdef Py_ssize_t alignment = (<unsigned int> &data[0]) % SIMD_ALIGNMENT
    cdef Py_ssize_t aligned_index = alignment / sizeof(double)

    cdef __m256d tmp_cum = _mm256_set1_pd(0)

    cdef __m256d tmp_sum
    for i in range(aligned_index, N-SIMD_LANES+1, SIMD_LANES):
        tmp_sum  = _mm256_load_pd(&data[i])
        tmp_cum  = _mm256_add_pd(tmp_cum, tmp_sum)

    cdef __m128d sum_high = _mm256_extractf128_pd(tmp_cum, 1)

    cdef __m128d simd_sum = _mm_add_pd(sum_high, _mm256_castpd256_pd128(tmp_cum))
    simd_sum = _mm_hadd_pd(simd_sum, simd_sum)

    cdef double mean = _mm_cvtsd_f64(simd_sum)

    for i in range(min(aligned_index, N)):
        mean += data[i]

    for i in range(N - ((N - aligned_index) % SIMD_LANES), N):
        mean += data[i]

    return mean / N

cdef double[:] mean_vec(double[:,:] data):
    # cdef double[:] return_single
    # if data.shape[1] == 1:
    #     return_single = np.empty((1,))
    #     return_single[0] = mean(data[:,0])

    # Non-contiguous data. Fallback to scalar code.
    if data.strides[0] != sizeof(double) and data.strides[1] != sizeof(double):
        return covariance_fallback.mean_vec(data)

    if data.strides[0] == sizeof(double):
        return mean_vec_columnmajor(data)
    else:
        return mean_vec_rowmajor(data)


cdef double[:] mean_vec_columnmajor(double[:,:] data):
    cdef Py_ssize_t N = data.shape[0], k = data.shape[1]
    cdef Py_ssize_t i, j, last_index
    cdef Py_ssize_t alignment, aligned_index

    cdef double[:] means = np.empty((k,))

    cdef __m256d tmp_cum, tmp_sum
    cdef __m128d sum_high, simd_sum

    cdef double mean

    for j in range(k):
        alignment = (<unsigned int> &data[0,k]) % SIMD_ALIGNMENT
        aligned_index = alignment / sizeof(double)

        tmp_cum = _mm256_set1_pd(0)

        for i in range(aligned_index, N-SIMD_LANES+1, SIMD_LANES):
            tmp_sum  = _mm256_load_pd(&data[i,j])
            tmp_cum  = _mm256_add_pd(tmp_cum, tmp_sum)

        sum_high = _mm256_extractf128_pd(tmp_cum, 1)

        simd_sum = _mm_add_pd(sum_high, _mm256_castpd256_pd128(tmp_cum))
        simd_sum = _mm_hadd_pd(simd_sum, simd_sum)

        mean = _mm_cvtsd_f64(simd_sum)

        for i in range(min(aligned_index, N)):
            mean += data[i,j]

        for i in range(N - ((N - aligned_index) % SIMD_LANES), N):
            mean += data[i,j]

        means[j] = mean / N

    return means


cdef double[:] mean_vec_rowmajor(double[:,:] data):
    sys.exit("Not implement row major mean_vec yet.")

cdef double sse(double[:] data, double mean):
    cdef Py_ssize_t N = data.shape[0], i

    # Non-contiguous data. Fallback to scalar code.
    if data.strides[0] != sizeof(double):
        return covariance_fallback.sse(data, mean)

    cdef int alignment = (<unsigned int> &data[0]) % SIMD_ALIGNMENT
    cdef int aligned_index = alignment / sizeof(double)

    cdef __m256d tmp_cum = _mm256_set1_pd(0)
    cdef __m256d mean_simd = _mm256_set1_pd(mean)

    cdef __m256d substract

    for i in range(aligned_index, N-SIMD_LANES+1, SIMD_LANES):
        substract  = _mm256_load_pd(&data[i])
        substract = _mm256_sub_pd(substract, mean_simd)
        tmp_cum = _mm256_add_pd(tmp_cum, _mm256_mul_pd(substract, substract))

    cdef __m128d sum_high = _mm256_extractf128_pd(tmp_cum, 1);
    cdef __m128d simd_sum = _mm_add_pd(sum_high, _mm256_castpd256_pd128(tmp_cum));
    simd_sum = _mm_hadd_pd(simd_sum, simd_sum)

    cdef double sse = _mm_cvtsd_f64(simd_sum)

    for i in range(min(aligned_index, N)):
        sse += (data[i] - mean) * (data[i] - mean)

    for i in range(N - ((N - aligned_index) % SIMD_LANES), N):
        sse += (data[i] - mean) * (data[i] - mean)

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