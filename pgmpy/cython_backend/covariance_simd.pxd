cdef extern from "x86intrin.h":
    ctypedef double __m256d
    ctypedef double __m128d
    __m256d _mm256_load_pd(const double* mem_addr) nogil
    __m256d _mm256_add_pd (__m256d a, __m256d b) nogil
    __m128d _mm256_extractf128_pd(__m256d a, const int imm8) nogil
    __m128d _mm_add_pd(__m128d a, __m128d b) nogil
    __m128d _mm256_castpd256_pd128 (__m256d a) nogil
    __m128d _mm_hadd_pd(__m128d a, __m128d b) nogil
    double _mm256_cvtsd_f64(__m256d a) nogil
    double _mm_cvtsd_f64(__m128d a) nogil
    void _mm_storeh_pd(double* mem_addr, __m128d a) nogil
    void _mm_storel_pd(double* mem_addr, __m128d a) nogil
    __m256d _mm256_set1_pd(double a) nogil
    __m256d _mm256_sub_pd(__m256d a, __m256d b) nogil
    __m256d _mm256_mul_pd(__m256d a, __m256d b) nogil

cdef double mean(double[:] data) nogil
cdef double[:] mean_vec(double[:,:] data)

cdef double sse(double[:] data, double mean)
cdef double[:,:] sse_mat(double[:,:] data, double[:] means)

cdef double[:,:] covariance(double[:,:] data, double[:] means)


cdef double[:,:] covariance_mat_with_vec(double[:,:] data_mat, double[:] data_vec, double[:] means_mat, double mean_vec)
cdef double[:,:] sse_mat_with_vec(double[:,:] data_mat, double[:] data_vec, double[:] means_mat, double mean_vec)

cdef double[:,:] drop_variable(double[:,:] cov, int index_to_drop)