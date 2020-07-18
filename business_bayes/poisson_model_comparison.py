import unittest

import numpy as np
import scipy
from scipy.stats import expon, poisson


class PoissonModels:
    """
    Compare two Poisson models for the same situation via calculating bayes factors

    Model 1: both counts depend on a single Poisson rate
        mu ~ Exponential(scale)
        counts1[i] ~ Poisson(mu)
        counts2[i] ~ Poisson(mu)

    Model 2: counts in each set depend on their own Poisson rate
        mu1 ~ Exponential(scale)
        mu2 ~ Exponential(scale)
        counts1[i] ~ Poisson(mu1)
        counts2[i] ~ Poisson(mu1)

    Parameters
    ----------

    scale :
        scale parameter for the exponential prior for Poisson rates
    """

    scale: float

    def __init__(self, scale):
        self.scale = scale

    def model_1_prob(self, counts1, counts2):
        def f(mu):
            ll = poisson.logpmf(counts1, mu).sum()
            ll += poisson.logpmf(counts2, mu).sum()
            ll += expon.logpdf(mu, self.scale)

            return np.exp(ll)

        return scipy.integrate.quad(f, 0, np.inf)

    def model_2_prob(self, counts1, counts2):

        gfun = lambda x: 0
        hfun = lambda x: np.inf

        def f(mu1, mu2):
            ll = poisson.logpmf(counts1, mu1).sum()
            ll += poisson.logpmf(counts2, mu2).sum()
            ll += expon.logpdf(mu1, self.scale)
            ll += expon.logpdf(mu2, self.scale)

            return np.exp(ll)

        return scipy.integrate.dblquad(f, 0, np.inf, gfun, hfun)

    def bayes_factor_bits(self, counts1, counts2):
        res_1 = self.model_1_prob(counts1, counts2)
        res_2 = self.model_2_prob(counts1, counts2)

        return np.log2(res_1[0]) - np.log2(res_2[0])


class Test(unittest.TestCase):
    def test_PoissonModels(self):
        b = PoissonModels(1)

        bits = b.bayes_factor_bits(np.array([50]), np.array([50]))
        self.assertTrue(bits > 3)

        bits = b.bayes_factor_bits(np.array([5]), np.array([3]))
        self.assertTrue(abs(bits) < 3)

        bits = b.bayes_factor_bits(np.array([11]), np.array([0]))
        self.assertTrue(bits < -1.6)

    def test_PoissonModels_array(self):
        b = PoissonModels(1)

        bits = b.bayes_factor_bits(np.array([50, 50, 50]), np.array([50]))
        self.assertTrue(bits > 3)

        bits = b.bayes_factor_bits(np.array([50, 30, 50]), np.array([50, 30, 50]))
        self.assertTrue(abs(bits) > 3)

        bits = b.bayes_factor_bits(np.array([11, 11]), np.array([0]))
        self.assertTrue(bits < -1.6)
