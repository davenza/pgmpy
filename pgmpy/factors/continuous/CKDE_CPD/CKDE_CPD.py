# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from six import string_types

from pgmpy.factors.base import BaseFactor
from pgmpy.factors.continuous.NodeType import NodeType

from scipy.special import logsumexp
from scipy.stats import norm, multivariate_normal

from ._ffi import ffi, lib
import atexit

from math import log, pi

import pandas as pd

from .ffi_helper import _CFFIDoubleArray, Error

class CKDE_CPD(BaseFactor):

    default_pro_que = None

    def __init__(self, variable, gaussian_cpds, kde_instances, evidence=None, evidence_type=None, bw_method=None):
        """
        Parameters
        ----------

        """
        if evidence is None:
            evidence = []
        if evidence_type is None:
            evidence_type = {}

        self.variable = variable
        self.evidence = evidence
        self.gaussian_evidence = []
        self.kde_evidence = []

        self.kde_indices = []

        for idx, e in enumerate(evidence):
            if evidence_type[e] == NodeType.GAUSSIAN:
                self.gaussian_evidence.append(e)
            elif evidence_type[e] == NodeType.SPBN:
                self.kde_evidence.append(e)
                self.kde_indices.append(idx+1)

        self.kde_indices = np.asarray(self.kde_indices, dtype=np.uint32)

        self.variables = [variable] + evidence

        self.gaussian_cpds = gaussian_cpds
        self.n_gaussian = len(gaussian_cpds)

        self.kde_instances = kde_instances.loc[:,[self.variable] + self.kde_evidence].to_numpy()

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

        kde_indices = ffi.cast("unsigned int*", self.kde_indices.ctypes.data)

        error = ffi.new("Error*", Error.NotFinished)

        lognorm_factor = self._logdenominator_factor()

        ckde = lib.ckde_init(self.pro_que,
                             self.kdedensity,
                             precision,
                             kde_indices,
                             self._ffi_gaussian_regressors,
                             len(self.gaussian_cpds),
                             lognorm_factor, error)

        if error[0] != Error.NoError:
            if error[0] == Error.MemoryError:
                raise MemoryError("Memory error allocating space in the OpenCL device.")
            elif error[0] == Error.NotFinished:
                raise Exception("CKDE code not finished.")

        self.ckde = ffi.gc(ckde, lib.ckde_free)

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

        error = ffi.new("Error*", Error.NotFinished)
        for idx, gaussian_cpd in enumerate(self.gaussian_cpds):
            evidence_index = np.asarray([self.evidence.index(e)+1 for e in gaussian_cpd.evidence if e != self.variable],
                                        dtype=np.uint32)

            evidence_index_ptr = ffi.cast("unsigned int*", evidence_index.ctypes.data)

            beta_ptr = ffi.cast("double*", gaussian_cpd.beta.ctypes.data)

            gr = lib.gaussian_regression_init(self.pro_que,
                                               self.evidence.index(gaussian_cpd.variable)+1,
                                               beta_ptr,
                                               evidence_index_ptr,
                                               len(gaussian_cpd.evidence),
                                               gaussian_cpd.variance,
                                               error)
            if error[0] != Error.NoError:
                if error[0] == Error.MemoryError:
                    raise MemoryError("Memory error allocating space in the OpenCL device.")
                elif error[0] == Error.NotFinished:
                    raise Exception("CKDE code not finished.")

            gr = ffi.gc(gr, lib.gaussian_regression_free)
            self._gaussian_regressors.append(gr)
            self._ffi_gaussian_regressors[idx] = gr

    def _init_kde(self):
        chol = _CFFIDoubleArray(self.cholesky, ffi)
        dataset = _CFFIDoubleArray(self.kde_instances, ffi)
        error = ffi.new("Error*", Error.NotFinished)
        self.kdedensity = lib.gaussian_kde_init(self.pro_que, chol.c_ptr(), dataset.c_ptr(), error)

        if error[0] != Error.NoError:
            if error[0] == Error.MemoryError:
                raise MemoryError("Memory error allocating space in the OpenCL device.")
            elif error[0] == Error.NotFinished:
                raise Exception("CKDE code not finished.")
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
        error = ffi.new("Error*", Error.NotFinished)
        lib.gaussian_kde_logpdf(self.kdedensity, self.pro_que, cffi_points.c_ptr(), cffi_result, error)
        if error[0] != Error.NoError:
            if error[0] == Error.MemoryError:
                raise MemoryError("Memory error allocating space in the OpenCL device.")
            elif error[0] == Error.NotFinished:
                raise Exception("CKDE code not finished.")
        return result


    def _logdenominator(self, instance):
        pass

    def logpdf_dataset(self, dataset):
        logpdf = self.kde_logpdf(dataset.loc[:,[self.variable] + self.kde_evidence])

        for i, gaussian_cpd in enumerate(self.gaussian_cpds):
            logpdf += gaussian_cpd.logpdf_dataset(dataset)

        logpdf -= self._logdenominator_dataset(dataset.loc[:,[self.variable] + self.evidence])

        return logpdf

    def _logdenominator_dataset(self, dataset):
        if self.n_kde == 0:
            if self.n_gaussian == 0:
                return np.zeros((len(dataset),))
            else:
                # TODO Sum in the GPU?
                return self._logdenominator_dataset_onlygaussian(dataset)
        else:
            if self.n_gaussian == 0:
                # TODO Sum in the GPU?
                return self._logdenominator_dataset_onlykde(dataset)
            else:
                # TODO Sum in the GPU?
                return self._logdenominator_dataset_mix(dataset)

    def _logdenominator_dataset_onlygaussian(self, dataset):
        points = np.atleast_2d(dataset.to_numpy())
        m, _ = points.shape

        result = np.empty((m,), dtype=np.float64)
        cffi_points = _CFFIDoubleArray(points, ffi)
        cffi_result = ffi.cast("double *", result.ctypes.data)
        error = ffi.new("Error*", Error.NotFinished)
        lib.logdenominator_dataset_gaussian(self.ckde, self.pro_que, cffi_points.c_ptr(), cffi_result, error)
        if error[0] != Error.NoError:
            if error[0] == Error.MemoryError:
                raise MemoryError("Memory error allocating space in the OpenCL device.")
            elif error[0] == Error.NotFinished:
                raise Exception("CKDE code not finished.")

        return result


    def _logdenominator_dataset_onlykde(self, dataset):
        points = np.atleast_2d(dataset.to_numpy())
        m, _ = points.shape

        result = np.empty((m,), dtype=np.float64)
        cffi_points = _CFFIDoubleArray(points, ffi)
        cffi_result = ffi.cast("double *", result.ctypes.data)
        error = ffi.new("Error*", Error.NotFinished)
        lib.logdenominator_dataset_onlykde(self.ckde, self.pro_que, cffi_points.c_ptr(), cffi_result, error)
        if error[0] != Error.NoError:
            if error[0] == Error.MemoryError:
                raise MemoryError("Memory error allocating space in the OpenCL device.")
            elif error[0] == Error.NotFinished:
                raise Exception("CKDE code not finished.")

        return result

    def _logdenominator_dataset_mix(self, dataset):
        points = np.atleast_2d(dataset.to_numpy())
        m, _ = points.shape

        result = np.empty((m,), dtype=np.float64)
        cffi_points = _CFFIDoubleArray(points, ffi)
        cffi_result = ffi.cast("double *", result.ctypes.data)
        error = ffi.new("Error*", Error.NotFinished)
        lib.logdenominator_dataset(self.ckde, self.pro_que, cffi_points.c_ptr(), cffi_result, error)
        if error[0] != Error.NoError:
            if error[0] == Error.MemoryError:
                raise MemoryError("Memory error allocating space in the OpenCL device.")
            elif error[0] == Error.NotFinished:
                raise Exception("CKDE code not finished.")
        return result


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

        cte = 0.5*np.log(np.pi) -\
              np.log(self.kde_instances.shape[0]) -\
              0.5*np.log(a) -\
              0.5*(self.n_gaussian + self.n_kde + 1)*np.log(2*np.pi) -\
              0.5*np.log(np.linalg.det(covariance)) -\
              logvar_mult

        prob = np.empty((dataset.shape[0],))

        for i in range(dataset.shape[0]):
            Ti = (dataset.iloc[i].loc[self.kde_evidence].to_numpy() - self.kde_instances[:,1:])

            bi = (self.kde_instances[:,0]*precision[0,0] - np.dot(Ti, precision[0, 1:]))

            ci = (np.sum(np.dot(Ti, precision[1:, 1:])*Ti, axis=1) -
                 2*self.kde_instances[:,0]*np.dot(Ti, precision[0, 1:]) +
                 (self.kde_instances[:,0]*self.kde_instances[:,0])*precision[0,0])

            for j, gaussian_cpd in enumerate(self.gaussian_cpds):
                k_index = gaussian_cpd.evidence.index(self.variable)

                Bjk = gaussian_cpd.beta[k_index+1]
                Gj = dataset.iloc[i, self.evidence.index(gaussian_cpd.variable)+1]

                subset_data = dataset.iloc[i].loc[gaussian_cpd.evidence].to_numpy()
                Cj = gaussian_cpd.beta[0] + np.dot(gaussian_cpd.beta[1:], subset_data) - Bjk*dataset.iloc[i].loc[self.variable]

                diff = Cj - Gj

                bi -= (Bjk * diff) / gaussian_cpd.variance
                ci += (diff * diff) / gaussian_cpd.variance

            ci *= -0.5
            prob[i] = logsumexp(((bi*bi) / (4*a)) + ci)

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
        df_kde = pd.DataFrame(self.kde_instances, columns=[self.variable] + self.kde_evidence)
        ev_type = {}

        for e in self.gaussian_evidence:
            ev_type[e] = NodeType.GAUSSIAN

        for e in self.kde_evidence:
            ev_type[e] = NodeType.SPBN

        copy_cpd = CKDE_CPD(self.variable, self.gaussian_cpds, df_kde, list(self.evidence), evidence_type=ev_type)
        return copy_cpd

    def __str__(self):
        pass

    def sample(self, N, parent_values):
        jcov = self.joint_cov_matrix()
        means = np.empty((self.n, len(self.evidence)))
        means[:, :self.n_kde] = self.kde_instances[:,1:]
        means[:,self.n_kde:] = self.means_gaussian(self.kde_instances)

        sampled = np.empty((N,))

        t = np.dot(jcov[0,1:], np.linalg.inv(jcov[1:,1:]))

        conditional_var = jcov[0,0] - t.dot(jcov[1:,0])

        marg_cov = jcov[1:, 1:]

        if not self.evidence:
            random_numbers = np.random.randint(low=0, high=self.n, size=N)

            sampled[:] = np.random.normal(self.kde_instances[random_numbers, 0],
                                          conditional_var)
        else:
            for i, (_, row) in enumerate(parent_values.iterrows()):
                r = row.loc[self.kde_evidence + self.gaussian_evidence].to_numpy()
                l = multivariate_normal.logpdf(means, r, marg_cov)

                weights = np.exp(l - logsumexp(l))
                cumsum_weights = np.cumsum(weights)

                random_number = np.random.uniform(size=1)
                index = np.digitize(random_number, cumsum_weights)[0]

                conditional_mean = self.kde_instances[index,0] + t.dot(r - means[index,:])
                sampled[i] = np.random.normal(conditional_mean, np.sqrt(conditional_var), size=1)

        return sampled

    def sample_weights(self, parent_values):
        jcov = self.joint_cov_matrix()
        means = np.empty((self.n, len(self.evidence)))
        means[:, :self.n_kde] = self.kde_instances[:,1:]
        means[:,self.n_kde:] = self.means_gaussian(self.kde_instances)


        weights = np.empty((self.n,))

        marg_cov = jcov[1:, 1:]

        if not self.evidence:

            weights[:] = 1/self.n
        else:
            for i, (_, row) in enumerate(parent_values.iterrows()):
                r = row.loc[self.kde_evidence + self.gaussian_evidence].to_numpy()
                l = multivariate_normal.logpdf(means, r, marg_cov)

                weights = np.exp(l - logsumexp(l))

        return means, weights

    def sample_distribution(self, domain, parent_values):
        jcov = self.joint_cov_matrix()
        means = np.empty((self.n, len(self.evidence)))
        means[:, :self.n_kde] = self.kde_instances[:,1:]
        means[:,self.n_kde:] = self.means_gaussian(self.kde_instances)

        t = np.dot(jcov[0,1:], np.linalg.inv(jcov[1:,1:]))
        conditional_var = jcov[0,0] - t.dot(jcov[1:,0])

        marg_cov = jcov[1:, 1:]

        r = parent_values.loc[self.kde_evidence + self.gaussian_evidence].to_numpy()
        l = multivariate_normal.logpdf(means, r, marg_cov)

        weights = np.exp(l - logsumexp(l))
        conditional_means = self.kde_instances[:,0] + t.dot((r - means).T)

        pdf = np.zeros_like(domain)
        for i, d in enumerate(domain):
            pdf[i] += np.sum(weights*norm.pdf(d, conditional_means, np.sqrt(conditional_var)))

        return pdf

    def joint_logpdf_dataset(self, dataset):
        logpdf = self.kde_logpdf(dataset.loc[:,[self.variable] + self.kde_evidence])

        for i, gaussian_cpd in enumerate(self.gaussian_cpds):
            logpdf += gaussian_cpd.logpdf_dataset(dataset)

        return logpdf

    def covariance_iy(self):
        cov_iy = np.empty((self.n_gaussian,))

        var_xi = self.covariance[0,0]
        for i, cpd in enumerate(self.gaussian_cpds):

            cov_iy[i] = cpd.beta[1+cpd.evidence.index(self.variable)]*var_xi

            for j, kde_parent in enumerate(self.kde_evidence):
                cov_iy[i] += cpd.beta[1+cpd.evidence.index(kde_parent)]*self.covariance[0,1+j]

            for k in range(i):
                cov_iy[i] += cpd.beta[1+cpd.evidence.index(self.gaussian_evidence[k])]*cov_iy[k]

        return cov_iy

    def covariance_zy(self):
        cov_zy = np.empty((self.n_kde, self.n_gaussian))

        for kde_index, kde_var in enumerate(self.kde_evidence):
            for i,cpd in enumerate(self.gaussian_cpds):

                cov_zy[kde_index, i] = cpd.beta[1+cpd.evidence.index(self.variable)]*\
                                                    self.covariance[0, 1+kde_index]

                for j, kde_ev_var in enumerate(self.kde_evidence):
                    cov_zy[kde_index, i] += cpd.beta[1+cpd.evidence.index(kde_ev_var)]*\
                                                        self.covariance[1+kde_index, 1+j]

                for k in range(i):
                    cov_zy[kde_index, i] += cpd.beta[1+cpd.evidence.index(self.gaussian_evidence[k])]*\
                                            cov_zy[kde_index, k]

        return cov_zy

    def linear_conditional_covariance(self):
        u = np.eye(self.n_gaussian)
        d = np.zeros((self.n_gaussian, self.n_gaussian))

        for cpd in self.gaussian_cpds:
            var_index = self.gaussian_evidence.index(cpd.variable)

            for ev_idx, ev in enumerate(cpd.evidence):
                if ev in self.gaussian_evidence:
                    ev_cpd_index = self.gaussian_evidence.index(ev)
                    u[var_index, ev_cpd_index] = -cpd.beta[1+ev_idx]

            d[var_index, var_index] = cpd.variance

        inv_u = np.linalg.inv(u)

        m = np.dot(inv_u, d).dot(inv_u.T)
        return m

    def means_gaussian(self, evidence):
        evidence_2d = np.atleast_2d(evidence)
        means = np.empty((evidence_2d.shape[0], self.n_gaussian))

        for i, cpd in enumerate(self.gaussian_cpds):
            means[:,i] = cpd.beta[0]

            ev_indices = 1 + np.asarray([cpd.evidence.index(self.variable)] +
                                        [cpd.evidence.index(e) for e in self.kde_evidence])
            means[:,i] += np.dot(cpd.beta[ev_indices], evidence_2d.T)

            if i > 0:
                previous_gaussians = [c.variable for c in self.gaussian_cpds[:i]]
                previous_gaussians_indices = 1 + np.asarray([cpd.evidence.index(p) for p in previous_gaussians])
                means[:,i] += np.dot(cpd.beta[previous_gaussians_indices], means[:,:i].T)

        return means

    def joint_cov_matrix(self):
        jcov = np.empty((len(self.variables), len(self.variables)))

        jcov[:self.d, :self.d] = self.covariance

        jcov[self.d:,0] = jcov[0, self.d:] = self.covariance_iy()

        jcov[1:self.d, self.d:] = self.covariance_zy()
        jcov[self.d:, 1:self.d] = jcov[1:self.d, self.d:].T

        regression_cond_cov = self.linear_conditional_covariance()
        s = jcov[self.d:, :self.d]
        jcov[self.d:, self.d:] = regression_cond_cov + np.dot(s, self.inv_cov).dot(s.T)

        return jcov

    def cond_logpdf_dataset(self, dataset):
        jcov = self.joint_cov_matrix()
        means = np.empty((self.n, len(self.variables)))

        logpdf = np.empty((dataset.shape[0],))

        r = np.dot(jcov[self.d:, :self.d], np.linalg.inv(jcov[:self.d, :self.d]))
        for i,(_, row) in enumerate(dataset.iterrows()):
            means[:, :self.d] = self.kde_instances

            d = row.loc[[self.variable] + self.kde_evidence].to_numpy() - self.kde_instances

            means[:,self.d:] = self.means_gaussian(row.loc[[self.variable] + self.kde_evidence]) - \
                               r.dot(d.T).T

            l = multivariate_normal.logpdf(means, row.loc[[self.variable] + self.kde_evidence + self.gaussian_evidence],
                                           jcov)

            l_marg = multivariate_normal.logpdf(means[:,1:], row.loc[self.kde_evidence + self.gaussian_evidence], jcov[1:,1:])

            logpdf[i] = logsumexp(l) - logsumexp(l_marg)

        return logpdf

    def conduni_logpdf_dataset(self, dataset):
        jcov = self.joint_cov_matrix()
        means = np.empty((self.n, len(self.variables)))

        logpdf = np.empty((dataset.shape[0],))

        t = np.dot(jcov[0,1:], np.linalg.inv(jcov[1:,1:]))
        r = np.dot(jcov[self.d:, :self.d], np.linalg.inv(jcov[:self.d, :self.d]))

        cond_var = jcov[0,0] - t.dot(jcov[1:, 0])
        for i,(_, row) in enumerate(dataset.iterrows()):
            means[:, :self.d] = self.kde_instances

            d = row.loc[[self.variable] + self.kde_evidence].to_numpy() - self.kde_instances

            means[:,self.d:] = self.means_gaussian(row.loc[[self.variable] + self.kde_evidence]) - \
                               r.dot(d.T).T


            p = row.loc[self.kde_evidence + self.gaussian_evidence].to_numpy() - means[:,1:]

            conditional_means = self.kde_instances[:,0] + t.dot(p.T)

            l = norm.logpdf(row.loc[self.variable], conditional_means, np.sqrt(cond_var))
            l_marg = multivariate_normal.logpdf(means[:,1:], row.loc[self.kde_evidence + self.gaussian_evidence], jcov[1:,1:])

            # w = l_marg - logsumexp(l_marg)
            logpdf[i] = logsumexp(l+l_marg) - logsumexp(l_marg)

        return logpdf

    def cond_joint_logpdf_dataset(self, dataset):
        jcov = self.joint_cov_matrix()
        means = np.empty((self.n, len(self.variables)))

        logpdf = np.empty((dataset.shape[0],))

        r = np.dot(jcov[self.d:, :self.d], np.linalg.inv(jcov[:self.d, :self.d]))
        for i,(_, row) in enumerate(dataset.iterrows()):
            means[:, :self.d] = self.kde_instances

            d = row.loc[[self.variable] + self.kde_evidence].to_numpy() - self.kde_instances

            means[:,self.d:] = self.means_gaussian(row.loc[[self.variable] + self.kde_evidence]) - \
                               r.dot(d.T).T

            l = multivariate_normal.logpdf(means, row.loc[[self.variable] + self.kde_evidence + self.gaussian_evidence],
                                           jcov)

            logpdf[i] = logsumexp(l) - np.log(self.n)

        return logpdf

    def conduni_joint_logpdf_dataset(self, dataset):
        jcov = self.joint_cov_matrix()
        means = np.empty((self.n, len(self.variables)))

        logpdf = np.empty((dataset.shape[0],))

        t = np.dot(jcov[0,1:], np.linalg.inv(jcov[1:,1:]))
        r = np.dot(jcov[self.d:, :self.d], np.linalg.inv(jcov[:self.d, :self.d]))

        cond_var = jcov[0,0] - t.dot(jcov[1:, 0])
        for i,(_, row) in enumerate(dataset.iterrows()):
            means[:, :self.d] = self.kde_instances

            d = row.loc[[self.variable] + self.kde_evidence].to_numpy() - self.kde_instances

            means[:,self.d:] = self.means_gaussian(row.loc[[self.variable] + self.kde_evidence]) - \
                               r.dot(d.T).T
            import pandas as pd
            kdf = pd.DataFrame(self.kde_instances, columns=[self.variable] + self.kde_evidence)
            self.means_gaussian(kdf.iloc[0,:])
            p = row.loc[self.kde_evidence + self.gaussian_evidence].to_numpy() - means[:,1:]

            conditional_means = self.kde_instances[:,0] + t.dot(p.T)

            l_cond = norm.logpdf(row.loc[self.variable], conditional_means, np.sqrt(cond_var))
            l_marg = multivariate_normal.logpdf(means[:,1:], row.loc[self.kde_evidence + self.gaussian_evidence], jcov[1:,1:])

            logpdf[i] = logsumexp(l_cond+l_marg) - np.log(self.n)

        return logpdf

    def __getstate__(self):
        state = self.__dict__.copy()

        del state['pro_que']
        del state['_ffi_gaussian_regressors']
        del state['_gaussian_regressors']
        del state['kdedensity']
        del state['ckde']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._initCFFI()