import unittest

import numpy as np
import scipy
from scipy.stats import expon, poisson


class ExponentialPrior:
    """
    Compare two Poisson models for the same situation via calculating bayes factors

    Model 1: both days depend on a single Poisson rate
        mu ~ Exponential(scale)
        day_1 ~ Poisson(mu)
        day_2 ~ Poisson(mu)

    Model 2: each day depends on its own Poisson rate
        mu1 ~ Exponential(scale)
        mu2 ~ Exponential(scale)
        day_1 ~ Poisson(mu1)
        day_2 ~ Poisson(mu2)

    Parameters
    ----------

    scale :
        scale parameter for the exponential prior for Poisson rates
    """

    scale: float

    def __init__(self, scale):
        self.scale = scale

    def model_1_prob(self, day_1, day_2):
        def f(mu):
            ll = poisson.logpmf(day_1, mu)
            ll += poisson.logpmf(day_2, mu)
            ll += expon.logpdf(mu, self.scale)

            return np.exp(ll)

        return scipy.integrate.quad(f, 0, np.inf)

    def model_2_prob(self, day_1, day_2):

        gfun = lambda x: 0
        hfun = lambda x: np.inf

        def f(mu1, mu2):
            ll = poisson.logpmf(day_1, mu1)
            ll += poisson.logpmf(day_2, mu2)
            ll += expon.logpdf(mu1, self.scale)
            ll += expon.logpdf(mu2, self.scale)

            return np.exp(ll)

        return scipy.integrate.dblquad(f, 0, np.inf, gfun, hfun)

    def bayes_factor_bits(self, day_1, day_2):
        res_1 = self.model_1_prob(day_1, day_2)
        res_2 = self.model_2_prob(day_1, day_2)

        return np.log2(res_1[0]) - np.log2(res_2[0])


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
    def test_exponential_prior(self):
        a = ExponentialPrior(1)

        bits = a.bayes_factor_bits(50, 50)
        print("bits (50, 50)", bits)
        self.assertTrue(bits > 3)

        bits = a.bayes_factor_bits(5, 3)
        print("bits (5, 3)", bits)
        self.assertTrue(abs(bits) < 3)

        bits = a.bayes_factor_bits(11, 0)
        print("bits (11, 0)", bits)
        self.assertTrue(bits < 3)

    def test_PoissonModels(self):
        b = PoissonModels(1)

        bits = b.bayes_factor_bits(np.array([50]), np.array([50]))
        self.assertTrue(bits > 3)

        bits = b.bayes_factor_bits(np.array([5]), np.array([3]))
        self.assertTrue(abs(bits) < 3)

        bits = b.bayes_factor_bits(np.array([11]), np.array([0]))
        self.assertTrue(bits < 3)
