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

cdef inline double sum_vector(double[:] v):
    cdef Py_ssize_t N = v.shape[0]
    cdef int alignment = (<unsigned int> &v[0]) % 32
    cdef int aligned_index = alignment / 8

    cdef __m256d tmp_cum = _mm256_load_pd(&v[aligned_index])

    cdef __m256d tmp_sum
    for i in range(aligned_index+4, N-4, 4):
        tmp_sum  = _mm256_load_pd(&v[i])
        tmp_cum  = _mm256_add_pd(tmp_cum, tmp_sum)

    cdef __m128d sum_high = _mm256_extractf128_pd(tmp_cum, 1);

    cdef __m128d result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(tmp_cum));
    cdef __m128d simd_sum = _mm_hadd_pd(result, result)

    cdef double double_sum = _mm_cvtsd_f64(simd_sum)

    cdef Py_ssize_t last_index = i
    for i in range(aligned_index):
        double_sum += v[i]

    for i in range(last_index+4, N):
        double_sum += v[i]

    return double_sum