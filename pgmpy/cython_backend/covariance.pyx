#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np

cdef extern from "x86intrin.h":
    ctypedef double __m256d
    ctypedef double __m128d
    __m256d _mm256_load_pd(const double* mem_addr)
    __m256d _mm256_add_pd (__m256d a, __m256d b)
    __m128d _mm256_extractf128_pd(__m256d a, const int imm8)
    __m128d _mm_add_pd(__m128d a, __m128d b)
    __m128d _mm256_castpd256_pd128 (__m256d a)
    __m128d _mm_hadd_pd(__m128d a, __m128d b)
    double _mm256_cvtsd_f64(__m256d a)
    double _mm_cvtsd_f64 (__m128d a)
    void _mm_storeh_pd (double* mem_addr, __m128d a)
    void _mm_storel_pd (double* mem_addr, __m128d a)


cdef double mean_slow(double[:] data):
    cdef double mean = 0
    cdef Py_ssize_t N = data.shape[0], i

    for i in range(N):
        mean += data[i]

    return mean / N

cdef extract_256d_components(__m256d s):
    cdef __m128d lower = _mm256_extractf128_pd(s, 0)
    cdef __m128d higher = _mm256_extractf128_pd(s, 1)

    cdef __m128d upper = _mm256_extractf128_pd(s, 1)

    cdef double first, second, third, fourth

    _mm_storel_pd(&first, lower)
    _mm_storeh_pd(&second, lower)

    _mm_storel_pd(&third, higher)
    _mm_storeh_pd(&fourth, higher)

    return (fourth, third, second, first)

cdef extract_128d_components(__m128d s):
    cdef double first, second

    _mm_storel_pd(&first, s)
    _mm_storeh_pd(&second, s)
    return (second, first)


cdef double mean(double[:] data):
    cdef double mean = 0
    cdef Py_ssize_t N = data.shape[0], i, j


    if data.strides[0] != 8:
        # print("Non contiguous memory stride " + str(data.strides[0]))
        return mean_slow(data)

    cdef double scalar_sum = 0
    for i in range(N):
        scalar_sum += data[i]

    print("Scalar sum " + str(scalar_sum))


    cdef int alignment = (<unsigned int> &data[0]) % 32

    print("Direction = " + str((<unsigned int> &data[0])))
    print("alignment = " + str(alignment))

    cdef int aligned_index = alignment / 8
    print("Aligned index = " + str(aligned_index))

    print("data[" + str(aligned_index) + "] is aligned " + str((<unsigned int> &data[aligned_index]) % 32))
    # if (<int> &data[alignment]) % 32 != 0:
        # print("Pointer not aligned")
        # print("New alignment " + str(alignment))

    # 20196.619434431374

    cdef __m256d tmp_cum = _mm256_load_pd(&data[aligned_index])

    cdef __m256d tmp_sum
    for i in range(aligned_index+4, N-4, 4):
        tmp_sum  = _mm256_load_pd(&data[i])
        tmp_cum  = _mm256_add_pd(tmp_cum, tmp_sum)

    cdef __m128d sum_high = _mm256_extractf128_pd(tmp_cum, 1);

    cdef __m128d result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(tmp_cum));
    cdef __m128d simd_sum = _mm_hadd_pd(result, result)

    cdef double double_sum = _mm_cvtsd_f64(simd_sum)

    for i in range(aligned_index):
        print("Summing " + str(i))
        double_sum += data[i]

    for i in range(aligned_index + int((N-aligned_index)/4)*4, N):
        print("Summing " + str(i))
        double_sum += data[i]

    print("scalar sum = " + str(scalar_sum))
    print("Simd sum = " + str(double_sum))


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
            cov[0,i+1] = (data_mat[m,i] - means_mat[i])*(data_vec[m] - mean_vec)

    for m in range(N):
        cov[0,0] += (data_vec[m] - mean_vec)

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
            sse[0,i+1] = (data_mat[m,i] - means_mat[i])*(data_vec[m] - mean_vec)

    for m in range(N):
        sse[0,0] += (data_vec[m] - mean_vec)

    for i in range(k+1):
        for j in range(i+1,k+1):
            sse[j,i] = sse[i,j]

    return sse

cdef double[:,:] drop_variable(double[:,:] cov, int index_to_drop):
    cdef Py_ssize_t k = cov.shape[0]

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