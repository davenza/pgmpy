# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from six import string_types

from pgmpy.factors.base import BaseFactor
from pgmpy.factors.continuous.NodeType import NodeType

from scipy.special import logsumexp

from ._ffi import ffi, lib
import atexit

from math import log, pi

from time import time

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

class CKDE_CPD(BaseFactor):

    default_pro_que = None

    def __init__(self, variable, gaussian_cpds, kde_instances, evidence=[], evidence_type={}, bw_method=None):
        """
        Parameters
        ----------

        """
        self.variable = variable
        self.evidence = evidence
        self.gaussian_evidence = []
        self.kde_evidence = []

        self.kde_indices = []

        for idx, e in enumerate(evidence):
            if evidence_type[e] == NodeType.GAUSSIAN:
                self.gaussian_evidence.append(e)
            elif evidence_type[e] == NodeType.CKDE:
                self.kde_evidence.append(e)
                self.kde_indices.append(idx+1)

        self.variables = [variable] + evidence

        self.gaussian_cpds = gaussian_cpds
        self.n_gaussian = len(gaussian_cpds)

        self.kde_instances = np.atleast_2d(kde_instances.loc[:,[self.variable] + self.kde_evidence].to_numpy())
        if self.kde_instances.shape[0] == 1:
            self.kde_instances = self.kde_instances.T

        self.n, self.d = self.kde_instances.shape

        self.set_bandwidth(bw_method=bw_method)

        self.n_kde = self.kde_instances.shape[1] - 1
        self._initCFFI()

    def scotts_factor(self):
        return np.power(self.n, -1. / (self.d + 4))

    def silverman_factor(self):
        return np.power(self.n * (self.d + 2.0) / 4.0, -1. / (self.d + 4))

    def set_bandwidth(self, bw_method=None):
        """
        Compute bandwidth
        :param bw_method: Method to compute bandwidth, as in scipy.stats.gaussian_kde.
        """
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, string_types):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance()

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = np.atleast_2d(np.cov(self.kde_instances, rowvar=False,
                                                         bias=False))
            self._data_inv_cov = np.linalg.inv(self._data_covariance)

        self.covariance = (self._data_covariance * self.factor**2)
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = np.sqrt(np.linalg.det(2*np.pi*self.covariance)) * self.n
        self.cholesky = np.linalg.cholesky(self.covariance)

    def _initCFFI(self, pro_que=None):
        if pro_que is None:
            if CKDE_CPD.default_pro_que is None:
                CKDE_CPD.default_pro_que = lib.new_proque()
                atexit.register(lib.gaussian_proque_free, CKDE_CPD.default_pro_que)

            self.pro_que = CKDE_CPD.default_pro_que
        else:
            self.pro_que = pro_que

        self._init_gaussian_regressors()
        self._init_kde()

        precision = ffi.cast("double*", self.inv_cov.ctypes.data)

        error = ffi.new("Error*", 0)

        lognorm_factor = self._logdenominator_factor()

        ckde = lib.ckde_init(self.pro_que,
                             self.kdedensity,
                             precision,
                             self._ffi_gaussian_regressors,
                             len(self.gaussian_cpds),
                             lognorm_factor, error)

        if error[0] == Error.MemoryError:
            raise MemoryError("Memory error allocating space in the OpenCL device.")

        self.ckde = ffi.gc(ckde, lib.ckde_free)
        # Remove the simple garbage finalizers.
        ffi.gc(self.kdedensity, None)
        for gr in self._gaussian_regressors:
            ffi.gc(gr, None)

    def _logdenominator_factor(self):
        s2 = 0
        logvar = 0
        for gaussian_cpd in self.gaussian_cpds:
            Bjk = gaussian_cpd.beta[gaussian_cpd.evidence.index(self.variable)+1]
            s2 += Bjk*Bjk / gaussian_cpd.variance
            logvar += 0.5*log(gaussian_cpd.variance)

        a = 0.5*(self.inv_cov[0,0] + s2)

        return 0.5*log(pi) - \
               log(self.n) - \
               0.5*log(a) - \
               0.5*(len(self.evidence) + 1)*log(2*pi) - \
               0.5*log(np.linalg.det(self.covariance)) - \
               logvar


    def _init_gaussian_regressors(self):
        self._ffi_gaussian_regressors = ffi.new("GaussianRegression*[]", self.n_gaussian)
        # Avoid freeing GaussianRegression before the CKDE is freed.
        self._gaussian_regressors = []

        error = ffi.new("Error*", 0)
        for idx, gaussian_cpd in enumerate(self.gaussian_cpds):
            evidence_index = np.asarray([self.evidence.index(e)+1 for e in gaussian_cpd.evidence if e != self.variable], dtype=np.uint32)

            evidence_index_ptr = ffi.cast("unsigned int*", evidence_index.ctypes.data)

            beta_ptr = ffi.cast("double*", gaussian_cpd.beta.ctypes.data)

            gr = lib.gaussian_regression_init(self.pro_que,
                                               self.evidence.index(gaussian_cpd.variable)+1,
                                               beta_ptr,
                                               evidence_index_ptr,
                                               len(gaussian_cpd.evidence),
                                               gaussian_cpd.variance,
                                               error)
            if error[0] == Error.MemoryError:
                raise MemoryError("Memory error allocating space in the OpenCL device.")

            gr = ffi.gc(gr, lib.gaussian_regression_free)
            self._gaussian_regressors.append(gr)
            self._ffi_gaussian_regressors[idx] = gr

    def _init_kde(self):
        chol = _CFFIDoubleArray(self.cholesky, ffi)
        dataset = _CFFIDoubleArray(self.kde_instances, ffi)
        error = ffi.new("Error*", 0)
        self.kdedensity = lib.gaussian_kde_init(self.pro_que, chol.c_ptr(), dataset.c_ptr(), error)
        if error[0] == Error.MemoryError:
            raise MemoryError("Memory error allocating space in the OpenCL device.")
        self.kdedensity = ffi.gc(self.kdedensity, lib.gaussian_kde_free)

    @property
    def pdf(self):
        def _pdf(*args):
            # TODO
            pass

        return _pdf

    def _denominator(self, instance):
        pass

    @property
    def logpdf(self):
        def _logpdf(*args):
            # TODO
            pass

        return _logpdf

    def kde_logpdf(self, dataset):
        points = np.atleast_2d(dataset.to_numpy())
        m, d = points.shape
        if d != self.d:
            if m == 1 == self.d:
                # points was passed in as a row vector
                points = points.T
                m = d
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d, self.d)
                raise ValueError(msg)

        result = np.empty((m,), dtype=np.float64)
        cffi_points = _CFFIDoubleArray(points, ffi)
        cffi_result = ffi.cast("double *", result.ctypes.data)
        error = ffi.new("Error*", 0)
        lib.gaussian_kde_logpdf(self.kdedensity, self.pro_que, cffi_points.c_ptr(), cffi_result, error)
        if error[0] == Error.MemoryError:
            raise MemoryError("Memory error allocating space in the OpenCL device.")
        return result


    def _logdenominator(self, instance):
        pass

    def logpdf_dataset(self, dataset):
        prob = 0
        for i, gaussian_cpd in enumerate(self.gaussian_cpds):
            prob += gaussian_cpd.logpdf_dataset(dataset)

        prob += self.kde_logpdf(dataset.loc[:,[self.variable] + self.kde_evidence]).sum()

        start = time()
        py_denominator = self._logdenominator_dataset_python(dataset[:10])
        end = time()
        py_time = end-start
        print("Python implementation: " + str(py_time))

        reorder_dataset = dataset.loc[:, [self.variable] + self.evidence]
        start = time()
        rust_denominator = self._logdenominator_dataset(reorder_dataset)[:10].sum()
        end = time()
        rust_time = end-start
        print("Rust implementation: " + str(rust_time))
        print("Ratio: " + str(py_time / rust_time))

        print("Python: " + str(py_denominator))
        print("Rust: " + str(rust_denominator))

        prob -= py_denominator

        return prob

    def _logdenominator_dataset(self, dataset):
        if self.n_kde == 0:
            if self.n_gaussian == 0:
                return 0
            else:
                return self._logdenominator_dataset_onlygaussian(dataset)
        else:
            if self.n_gaussian == 0:
                return self._logdenominator_dataset_onlykde(dataset)
            else:
                return self._logdenominator_dataset_mix(dataset)

    def _logdenominator_dataset_onlygaussian(self, dataset):
        points = np.atleast_2d(dataset.to_numpy())
        m, _ = points.shape

        result = np.empty((m,), dtype=np.float64)
        cffi_points = _CFFIDoubleArray(points, ffi)
        cffi_result = ffi.cast("double *", result.ctypes.data)
        error = ffi.new("Error*", 0)
        lib.logdenominator_dataset_gaussian(self.ckde, self.pro_que, cffi_points.c_ptr(), cffi_result, error)
        if error[0] == Error.MemoryError:
            raise MemoryError("Memory error allocating space in the OpenCL device.")
        return result


    def _logdenominator_dataset_onlykde(self, dataset):
        points = np.atleast_2d(dataset.to_numpy())
        m, _ = points.shape

        result = np.empty((m,), dtype=np.float64)
        cffi_points = _CFFIDoubleArray(points, ffi)
        cffi_result = ffi.cast("double *", result.ctypes.data)
        error = ffi.new("Error*", 0)
        lib.logdenominator_dataset_onlykde(self.ckde, self.pro_que, cffi_points.c_ptr(), cffi_result, error)
        if error[0] == Error.MemoryError:
            raise MemoryError("Memory error allocating space in the OpenCL device.")
        return result

    def _logdenominator_dataset_mix(self, dataset):
        pass

    def _logdenominator_dataset_python(self, dataset):
        dataset = dataset.loc[:, [self.variable] + self.evidence]

        covariance = self.covariance
        precision = self.inv_cov

        logvar_mult = 0
        a = precision[0,0]
        for gaussian_cpd in self.gaussian_cpds:
            k_index = gaussian_cpd.evidence.index(self.variable)
            a += (gaussian_cpd.beta[k_index+1] * gaussian_cpd.beta[k_index+1]) / gaussian_cpd.variance
            logvar_mult += 0.5*np.log(gaussian_cpd.variance)

        a *= 0.5

        cte = dataset.shape[0]*(0.5*np.log(np.pi) -
              np.log(self.kde_instances.shape[0]) -
              0.5*np.log(a) -
              0.5*(self.n_gaussian + self.n_kde + 1)*np.log(2*np.pi) -
              0.5*np.log(np.linalg.det(covariance)) -
              logvar_mult)

        prob = 0

        print("Python Implementation")
        print("------------------------")

        s1 = np.zeros((dataset.shape[0],))
        s3 = np.zeros((dataset.shape[0],))
        Cj_arr = np.zeros((dataset.shape[0],))
        Gj_arr = np.zeros((dataset.shape[0],))
        final = np.zeros((dataset.shape[0],))
        max_array = np.zeros((dataset.shape[0],))

        print("dataset =")
        print(dataset)
        for i in range(dataset.shape[0]):
            Ti = (dataset.iloc[i, [0] + self.kde_indices].values - self.kde_instances)

            bi = (self.kde_instances[:,0]*precision[0,0] - np.dot(Ti[:,1:], precision[0, 1:]))

            ci = (np.sum(np.dot(Ti[:,1:], precision[1:, 1:])*Ti[:,1:], axis=1) -
                 2*self.kde_instances[:,0]*np.dot(Ti[:,1:], precision[0, 1:]) +
                 (self.kde_instances[:,0]*self.kde_instances[:,0])*precision[0,0])

            for j, gaussian_cpd in enumerate(self.gaussian_cpds):
                k_index = gaussian_cpd.evidence.index(self.variable)

                Bjk = gaussian_cpd.beta[k_index+1]
                Gj = dataset.iloc[i, self.evidence.index(gaussian_cpd.variable)+1]

                subset_data = dataset.loc[i, gaussian_cpd.evidence].values
                Cj = gaussian_cpd.beta[0] + np.dot(gaussian_cpd.beta[1:], subset_data) - Bjk*dataset.loc[i,self.variable]

                # for b, s in zip(gaussian_cpd.beta[1:], subset_data):
                #     print("Adding beta " + str(b) + " * test " + str(s))
                # print("Substracting beta " + str(Bjk) + " * test " + str(dataset.loc[i,self.variable]))


                # print("beta0 = " + str(gaussian_cpd.beta[0]))
                # print("Cj = " + str(Cj))
                diff = Cj - Gj


                bi -= (Bjk * diff) / gaussian_cpd.variance
                ci += (diff * diff) / gaussian_cpd.variance

                s1[i] += (Bjk * diff) / gaussian_cpd.variance
                # print("s1 partial = " + str(s1))
                s3[i] += diff*diff / gaussian_cpd.variance
                # print("s3 partial = " + str(s3))

                Cj_arr[i] = Cj
                Gj_arr[i] = Gj

            ci *= -0.5

            max_array[i] = np.max(((bi*bi) / (4*a)) + ci)
            final[i] = logsumexp(((bi*bi) / (4*a)) + ci)
            prob += logsumexp(((bi*bi) / (4*a)) + ci)

        print("max = " + str(max_array[:10]))
        print("final = " + str(final))
        return cte + prob

    def copy(self):
        """
        Returns a copy of the distribution.

        Returns
        -------
        LinearGaussianCPD: copy of the distribution

        Examples
        --------
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> cpd = LinearGaussianCPD('Y',  [0.2, -2, 3, 7], 9.6, ['X1', 'X2', 'X3'])
        >>> copy_cpd = cpd.copy()
        >>> copy_cpd.variable
        'Y'
        >>> copy_cpd.evidence
        ['X1', 'X2', 'X3']
        """
        # TODO: Is this a real copy?
        copy_cpd = CKDE_CPD(self.variable, self.gaussian_cpds, self.kde_instances, list(self.evidence))
        return copy_cpd

    def __str__(self):
        pass