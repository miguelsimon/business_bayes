import unittest

import numpy as np
import scipy
from scipy.stats import poisson


class PoissonModels:
    """
    Compare two Poisson models for the same situation via calculating bayes factors

    Model 1: both counts depend on a single Poisson rate
        mu ~ LogUniform(a, b)
        counts1[i] ~ Poisson(mu)
        counts2[i] ~ Poisson(mu)

    Model 2: counts in each set depend on their own Poisson rate
        mu1 ~ LogUniform(a, b)
        mu2 ~ LogUniform(a, b)
        counts1[i] ~ Poisson(mu1)
        counts2[i] ~ Poisson(mu1)

    Parameters
    ----------
    a :
        lower bound on rate
    b :
        upper bound on rate
    """

    a: float
    b: float

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def model_1_prob(self, counts1, counts2):
        a = self.a
        b = self.b

        constant = np.log(b) - np.log(a)

        def f(x):
            ll = poisson.logpmf(counts1, x).sum()
            ll += poisson.logpmf(counts2, x).sum()
            ll -= np.log(x)

            return np.exp(ll)

        return scipy.integrate.quad(f, a, b) / constant

    def model_2_prob(self, counts1, counts2):
        a = self.a
        b = self.b

        constant = np.log(b) - np.log(a)

        def f1(x):
            ll = poisson.logpmf(counts1, x).sum()
            ll -= np.log(x)
            return np.exp(ll)

        def f2(x):
            ll = poisson.logpmf(counts2, x).sum()
            ll -= np.log(x)
            return np.exp(ll)

        term1 = scipy.integrate.quad(f1, a, b)
        term2 = scipy.integrate.quad(f2, a, b)

        res = term1[0] * term2[0] / (constant * constant)
        err = max(term1[1], term2[1]) / (constant * constant)

        return [res, err]

    def bayes_factor_bits(self, counts1, counts2):
        res_1 = self.model_1_prob(counts1, counts2)
        res_2 = self.model_2_prob(counts1, counts2)

        return np.log2(res_1[0]) - np.log2(res_2[0])


class Test(unittest.TestCase):
    def test_PoissonModels(self):
        b = PoissonModels(1 / 100, 100)

        bits = b.bayes_factor_bits(np.array([50]), np.array([50]))
        self.assertTrue(bits > 3)

        bits = b.bayes_factor_bits(np.array([5]), np.array([3]))
        self.assertTrue(abs(bits) < 3)

        bits = b.bayes_factor_bits(np.array([11]), np.array([0]))
        self.assertTrue(bits < -1.6)

    def test_PoissonModels_array(self):
        b = PoissonModels(1 / 100, 100)

        bits = b.bayes_factor_bits(np.array([50, 50, 50]), np.array([50]))
        self.assertTrue(bits > 3)

        bits = b.bayes_factor_bits(np.array([50, 30, 50]), np.array([50, 30, 50]))
        self.assertTrue(abs(bits) > 3)

        bits = b.bayes_factor_bits(np.array([11, 11]), np.array([0]))
        self.assertTrue(bits < -1.6)
