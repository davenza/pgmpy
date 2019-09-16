cdef void inverse(double[:,:] mat, double[:,:] out_inv)

cdef void inverse_symmetric_psd(double[:,:] mat, double[:,:] out_inv)
cdef void dgemv(double alpha, double[:,:] A, double[:] x, double beta, double[:] y)

cdef double det(double[:,:] mat, bint overwrite=*)