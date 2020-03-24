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

from .ffi_helper import _CFFIDoubleArray, Error


class ConditionalKDE(BaseFactor):

    default_pro_que = None

    def __init__(self, variable, training_data, evidence=None, bw_method=None):
        """
        Parameters
        ----------

        """
        if evidence is None:
            evidence = []

        self.variable = variable
        self.evidence = evidence
        self.variables = [variable] + evidence

        # FIXME: Hay que hacer copia?
        self._kde_joint_data = training_data.loc[:, self.variables].to_numpy()
        self._kde_marg_data = training_data.loc[:, self.evidence].to_numpy()
        self.n, self.d = self._kde_joint_data.shape
        self.set_bandwidth(bw_method=bw_method)
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
            self._joint_data_covariance = np.atleast_2d(np.cov(self._kde_joint_data, rowvar=False,
                                                         bias=False))
            self._joint_data_inv_cov = np.linalg.inv(self._joint_data_covariance)

        self.joint_covariance = (self._joint_data_covariance * self.factor**2)
        self.joint_inv_cov = self._joint_data_inv_cov / self.factor**2
        self.joint_norm_factor = np.sqrt(np.linalg.det(2*np.pi*self.joint_covariance)) * self.n
        self.joint_cholesky = np.linalg.cholesky(self.joint_covariance)

        self.marg_covariance = self.joint_covariance[1:,1:]
        self.marg_cholesky = np.linalg.cholesky(self.marg_covariance)

    def _initCFFI(self, pro_que=None):
        if pro_que is None:
            if ConditionalKDE.default_pro_que is None:
                ConditionalKDE.default_pro_que = lib.new_proque()
                atexit.register(lib.gaussian_proque_free, ConditionalKDE.default_pro_que)

            self.pro_que = ConditionalKDE.default_pro_que
        else:
            self.pro_que = pro_que

        self._init_kde()

    def _init_kde(self):
        joint_chol = _CFFIDoubleArray(self.joint_cholesky, ffi)
        joint_dataset = _CFFIDoubleArray(self._kde_joint_data, ffi)
        error = ffi.new("Error*", Error.NotFinished)

        self.joint_kdedensity = lib.gaussian_kde_init(self.pro_que, joint_chol.c_ptr(), joint_dataset.c_ptr(), error)

        if error[0] != Error.NoError:
            if error[0] == Error.MemoryError:
                raise MemoryError("Memory error allocating space in the OpenCL device.")
            elif error[0] == Error.NotFinished:
                raise Exception("ConditionalKDE code not finished.")

        error[0] = Error.NotFinished
        self.joint_kdedensity = ffi.gc(self.joint_kdedensity, lib.gaussian_kde_free)

        if self.evidence:
            marg_chol = _CFFIDoubleArray(self.marg_cholesky, ffi)
            marg_dataset = _CFFIDoubleArray(self._kde_marg_data, ffi)
            self.marg_kdedensity = lib.gaussian_kde_init(self.pro_que, marg_chol.c_ptr(), marg_dataset.c_ptr(), error)

            if error[0] != Error.NoError:
                if error[0] == Error.MemoryError:
                    raise MemoryError("Memory error allocating space in the OpenCL device.")
            elif error[0] == Error.NotFinished:
                raise Exception("ConditionalKDE code not finished.")

            self.marg_kdedensity = ffi.gc(self.marg_kdedensity, lib.gaussian_kde_free)

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

    def conditional_logpdf(self, dataset):
        points_joint = np.atleast_2d(dataset.to_numpy())
        m, d = points_joint.shape
        if d != self.d:
            if m == 1 == self.d:
                # points was passed in as a row vector
                points_joint = points_joint.T
                m = d
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d, self.d)
                raise ValueError(msg)

        result_joint = np.empty((m,), dtype=np.float64)

        cffi_joint_points = _CFFIDoubleArray(points_joint, ffi)
        cffi_joint_result = ffi.cast("double *", result_joint.ctypes.data)

        error = ffi.new("Error*", Error.NotFinished)
        lib.gaussian_kde_logpdf(self.joint_kdedensity, self.pro_que, cffi_joint_points.c_ptr(), cffi_joint_result, error)
        if error[0] != Error.NoError:
            if error[0] == Error.MemoryError:
                raise MemoryError("Memory error allocating space in the OpenCL device.")
            elif error[0] == Error.NotFinished:
                raise Exception("CKDE code not finished.")

        if self.evidence:
            points_marg = np.atleast_2d(dataset.loc[:, self.evidence].to_numpy())
            m, d = points_marg.shape
            if d != self.d-1:
                if m == 1 == self.d-1:
                    # points was passed in as a row vector
                    points_marg = points_marg.T
                    m = d
                else:
                    msg = "points have dimension %s, dataset has dimension %s" % (d, self.d)
                    raise ValueError(msg)

            result_marg = np.empty((m,), dtype=np.float64)

            cffi_marg_points = _CFFIDoubleArray(points_marg, ffi)
            cffi_marg_result = ffi.cast("double *", result_marg.ctypes.data)

            error = ffi.new("Error*", Error.NotFinished)
            lib.gaussian_kde_logpdf(self.marg_kdedensity, self.pro_que, cffi_marg_points.c_ptr(), cffi_marg_result, error)
            if error[0] != Error.NoError:
                if error[0] == Error.MemoryError:
                    raise MemoryError("Memory error allocating space in the OpenCL device.")
                elif error[0] == Error.NotFinished:
                    raise Exception("CKDE code not finished.")

            return result_joint - result_marg
        else:
            return result_joint


    def logpdf_dataset(self, dataset):
        return self.conditional_logpdf(dataset.loc[:,self.variables])

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
        # TODO: Review this copy
        copy_cpd = ConditionalKDE(self.variable, self._kde_joint_data, list(self.evidence))
        return copy_cpd

    def __str__(self):
        pass