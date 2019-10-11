# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import pandas as pd
from scipy.stats import norm

from pgmpy.factors.base import BaseFactor
from pgmpy.models import HybridContinuousModel

from scipy.special import logsumexp

from ._ffi import ffi, lib
import atexit
from kde_ocl import gaussian_kde_ocl

class Error:
    NoError = 0
    MemoryError = 1

class CKDE_CPD(BaseFactor):

    default_pro_que = None

    def __init__(self, variable, gaussian_cpds, kde_instances, evidence=[], evidence_type={}):
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
            if evidence_type[e] == HybridContinuousModel.NodeType.GAUSSIAN:
                self.gaussian_evidence.append(e)
            elif evidence_type[e] == HybridContinuousModel.NodeType.CKDE:
                self.kde_evidence.append(e)
                self.kde_indices.append(idx+1)

        self.variables = [variable] + evidence

        self.gaussian_cpds = gaussian_cpds
        self.n_gaussian = len(gaussian_cpds)

        self.gaussian_indices = []

        for gaussian_cpd in self.gaussian_cpds:
            tmp = []
            tmp.append(self.evidence.index(gaussian_cpd.variable) + 1)
            for e in gaussian_cpd.evidence:
                if e != variable:
                    tmp.append(self.evidence.index(e) + 1)
                else:
                    tmp.append(0)

            self.gaussian_indices.append(tmp)

            # self.gaussian_indices.append([self.evidence.index(e) + 1 if e != variable else 0 for e in gaussian_cpd.evidence])

        self.kde_instances = kde_instances.values
        self.joint_kde = gaussian_kde_ocl(self.kde_instances)
        self.n_kde = self.kde_instances.shape[1] - 1

        self._initCFFI()

        super(CKDE_CPD, self).__init__()

    def _initCFFI(self, pro_que=None):
        if pro_que is None:
            if gaussian_kde_ocl.default_pro_que is not None:
                self.pro_que = gaussian_kde_ocl.default_pro_que
            else:
                if CKDE_CPD.default_pro_que is None:
                    CKDE_CPD.default_pro_que = lib.new_proque()
                    atexit.register(lib.gaussian_proque_free, gaussian_kde_ocl.default_pro_que)

                self.pro_que = CKDE_CPD.default_pro_que
        else:
            self.pro_que = pro_que


        self._init_gaussian_regressors()

        precision = ffi.cast("double*", self.joint_kde.inv_cov.ctypes.data)

        error = ffi.new("Error*", 0)
        ckde = lib.ckde_init(self.pro_que, self.joint_kde.kdedensity, precision, self._ffi_gaussian_regressors, len(self.gaussian_cpds))
        if error[0] == Error.MemoryError:
            raise MemoryError("Memory error allocating space in the OpenCL device.")

        self.ckde = ffi.gc(ckde, lib.ckde_free)
        pass

    def _init_gaussian_regressors(self):
        self._ffi_gaussian_regressors = ffi.new("GaussianRegression[]", self.n_gaussian)

        for idx, gaussian_cpd in enumerate(self.gaussian_cpds):
            evidence_index = np.empty((len(gaussian_cpd.evidence),), dtype=np.int)

            for ev_idx, e in enumerate(gaussian_cpd.evidence):
                evidence_index[ev_idx] = self.evidence.index(e)

            evidence_index_ptr = ffi.cast("int*", evidence_index.ctypes.data)
            beta_ptr = ffi.cast("double*", gaussian_cpd.beta.ctypes.data)

            error = ffi.new("Error*", 0)
            gr = lib.gaussian_regression_init(self.pro_que,
                                               self.evidence.index(gaussian_cpd.variable),
                                               beta_ptr,
                                               evidence_index_ptr,
                                               len(gaussian_cpd.evidence),
                                               gaussian_cpd.variance,
                                               error)
            if error[0] == Error.MemoryError:
                raise MemoryError("Memory error allocating space in the OpenCL device.")
            self._ffi_gaussian_regressors[idx] = ffi.gc(gr, lib.gaussian_regression_free)

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

    def _logdenominator(self, instance):
        pass

    def logpdf_dataset(self, dataset):
        prob = 0
        for i, gaussian_cpd in enumerate(self.gaussian_cpds):
            indices = self.gaussian_indices[i]
            subset_data = dataset.iloc[:, indices]
            prob += gaussian_cpd.logpdf_dataset(subset_data)

        prob += self.joint_kde.logpdf(dataset.loc[:,[self.variable] + self.kde_evidence]).sum()

        prob -= self._logdenominator_dataset_python(dataset)

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
        logvar_mult = 0

        a = 0

        bi = self.kde_instances[:,0] * self.joint_kde.inv_cov[0,0]
        ci = self.kde_instances[:,0] * self.kde_instances[:,0] * self.joint_kde.inv_cov[0,0]
        for j, gaussian_cpd in enumerate(self.gaussian_cpds):
            k_index = gaussian_cpd.evidence.index(self.variable)

            Bjk = gaussian_cpd.beta[k_index+1]
            Gj = dataset.iloc[:, self.evidence.index(gaussian_cpd.variable)+1]

            indices = self.gaussian_indices[j]
            subset_data = dataset.iloc[:, indices].values
            Cj = gaussian_cpd.beta[0] + np.sum(gaussian_cpd.beta[None, 1:]*subset_data[:,1:], axis=1) - Bjk*dataset.iloc[:,0]

            diff = Cj - Gj

            bi -= ((Bjk * diff) / gaussian_cpd.variance).sum()

            ci += ((diff * diff) / gaussian_cpd.variance).sum()

            a += (Bjk * Bjk) / gaussian_cpd.variance
            logvar_mult += np.log(np.sqrt(gaussian_cpd.variance))

        ci *= -0.5

        print("Only gaussian")
        print("------------------------")
        # print("Cj = " + str(Cj))
        # print("a = " + str(a))
        print("bi = " + str(bi))
        # print("ci = " + str(ci))
        res = dataset.shape[0]*(0.5*np.log(np.pi) -
              np.log(self.kde_instances.shape[0]) -
              0.5*np.log(a) -
              0.5*(self.n_gaussian + 1)*np.log(2*np.pi) -
              0.5*np.log(self.joint_kde.covariance[0,0]) -
              logvar_mult)

        res += logsumexp(((bi*bi) / (4*a)) + ci)
        return res

    def _logdenominator_dataset_onlykde(self, dataset):

        a = 0.5*self.joint_kde.inv_cov[0,0]
        cte = dataset.shape[0]*(0.5*np.log(np.pi) -
                                np.log(self.kde_instances.shape[0]) -
                                0.5*np.log(a) -
                                0.5*(self.n_kde + 1)*np.log(2*np.pi) -
                                0.5*np.log(np.linalg.det(self.joint_kde.covariance))
                                )

        return cte + self.joint_kde.denominator_onlykde(dataset.values)

    def _logdenominator_dataset_mix(self, dataset):

        logvar_mult = 0
        a_const = 0
        b_const = 0
        c_const = 0

        for j, gaussian_cpd in enumerate(self.gaussian_cpds):
            k_index = gaussian_cpd.evidence.index(self.variable)

            Bjk = gaussian_cpd.beta[k_index+1]
            Gj = dataset.iloc[:, self.evidence.index(gaussian_cpd.variable)+1]

            indices = self.gaussian_indices[j]
            subset_data = dataset.iloc[:, indices].values
            Cj = gaussian_cpd.beta[0] + np.dot(gaussian_cpd.beta[1:], subset_data[:,1:]) - Bjk*dataset[:,0]

            diff = Cj - Gj

            b_const += (Bjk * diff) / gaussian_cpd.variance

            c_const += (diff * diff) / gaussian_cpd.variance

            a_const += (Bjk * Bjk) / gaussian_cpd.variance
            logvar_mult += np.log(np.sqrt(gaussian_cpd.variance))

        precision = self.joint_kde.inv_cov
        a = 0.5*(precision[0,0] + a_const)
        cte = dataset.shape[0]*(0.5*np.log(np.pi) -
                                np.log(self.kde_instances.shape[0]) -
                                0.5*np.log(a) -
                                0.5*(self.n_kde + self.n_gaussian + 1)*np.log(2*np.pi) -
                                0.5*np.log(np.linalg.det(self.joint_kde.covariance)) -
                                logvar_mult)


        return cte + self.joint_kde.denominator(dataset.values, a_const, b_const, c_const)

    def _logdenominator_dataset_python(self, dataset):
        covariance = self.joint_kde.covariance
        precision = self.joint_kde.inv_cov

        logvar_mult = 0
        a = precision[0,0]
        for gaussian_cpd in self.gaussian_cpds:
            k_index = gaussian_cpd.evidence.index(self.variable)
            a += (gaussian_cpd.beta[k_index+1] * gaussian_cpd.beta[k_index+1]) / gaussian_cpd.variance
            logvar_mult += np.log(np.sqrt(gaussian_cpd.variance))

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

                indices = self.gaussian_indices[j]
                subset_data = dataset.iloc[i, indices].values
                Cj = gaussian_cpd.beta[0] + np.dot(gaussian_cpd.beta[1:], subset_data[1:]) - Bjk*dataset.iloc[i,0]

                diff = Cj - Gj

                bi -= (Bjk * diff) / gaussian_cpd.variance

                ci += (diff * diff) / gaussian_cpd.variance



            ci *= -0.5
            print("Iteration " + str(i))
            # print("Cj = " + str(Cj))
            # print("a = " + str(a))
            print("bi = " + str(bi))
            # print("ci = " + str(ci))
            prob += logsumexp(((bi*bi) / (4*a)) + ci)

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