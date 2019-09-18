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


cdef extract_256d_components(__m256d s)

cdef extract_128d_components(__m128d s)

cdef double sum_vector(double[:] v)