import unittest

import numpy as np
import scipy
from scipy.stats import binom


class Binomials:
    """
    Calculate bayes factor for two models, one with a single p parameter, one
    with p1 and p2 parameters, for 2 different binomial experiments:

    * the first experiment yields count_1 successes in trials_1 trials
    * the second experiment yields count_2 successes in trials_2 trials

    Model 1: both depend on same p
        p ~ Uniform(0, 1)
        count_1 ~ Binomial(trials_1, p)
        count_2 ~ Binomial(trials_2, p)

    Model 2: different ps
        p1 ~ Uniform(0, 1)
        p2 ~ Uniform(0, 1)
        count_1 ~ Binomial(trials_1, p1)
        count_2 ~ Binomial(trials_2, p2)

    """

    def model_1_prob(self, count_1, trials_1, count_2, trials_2):
        def f(p):
            return binom.pmf(count_1, trials_1, p) * binom.pmf(count_2, trials_2, p)

        return scipy.integrate.quad(f, 0, 1)

    def model_2_prob(self, count_1, trials_1, count_2, trials_2):
        def f1(p):
            return binom.pmf(count_1, trials_1, p)

        def f2(p):
            return binom.pmf(count_2, trials_2, p)

        prob1, err1 = scipy.integrate.quad(f1, 0, 1)
        prob2, err2 = scipy.integrate.quad(f2, 0, 1)

        return prob1 * prob2, prob1 * err2 + prob2 * err1 + err1 * err2

    def bayes_factor_bits(self, count_1, trials_1, count_2, trials_2):
        res_1 = self.model_1_prob(count_1, trials_1, count_2, trials_2)
        res_2 = self.model_2_prob(count_1, trials_1, count_2, trials_2)

        return np.log2(res_1[0]) - np.log2(res_2[0])


class Test(unittest.TestCase):
    def test(self):
        a = Binomials()

        bits = a.bayes_factor_bits(1, 100, 1, 100)
        self.assertTrue(bits > 3)

        bits = a.bayes_factor_bits(50, 100, 0, 100)
        self.assertTrue(bits < 3)
