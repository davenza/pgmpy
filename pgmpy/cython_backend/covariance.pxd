cdef double mean(double[:] data)
cdef double[:] mean_vec(double[:,:] data)

cdef double sse(double[:] data, double mean)
cdef double[:,:] sse_mat(double[:,:] data, double[:] means)

cdef double[:,:] covariance(double[:,:] data, double[:] means)


cdef double[:,:] covariance_mat_with_vec(double[:,:] data_mat, double[:] data_vec, double[:] means_mat, double mean_vec)
cdef double[:,:] sse_mat_with_vec(double[:,:] data_mat, double[:] data_vec, double[:] means_mat, double mean_vec)

cdef double[:,:] drop_variable(double[:,:] cov, int index_to_drop)