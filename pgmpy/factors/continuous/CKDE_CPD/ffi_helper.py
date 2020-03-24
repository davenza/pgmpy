from ._ffi import ffi, lib

class _CFFIDoubleArray(object):
    def __init__(self, array, ffi):
        self.shape = ffi.new("size_t[]", array.shape)
        self.strides = ffi.new("size_t[]", array.strides)
        self.arrayptr = ffi.cast("double*", array.ctypes.data)
        self.cffiarray = ffi.new('DoubleNumpyArray*', {'ptr': self.arrayptr,
                                                       'size': array.size,
                                                       'ndim': array.ndim,
                                                       'shape': self.shape,
                                                       'strides': self.strides})
    def c_ptr(self):
        return self.cffiarray

class Error:
    NoError = 0
    MemoryError = 1
    NotFinished = 2