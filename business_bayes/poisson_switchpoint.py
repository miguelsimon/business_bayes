import unittest

import numpy as np
import pymc3 as pm
import scipy
from numpy import ndarray


def make_switchpoint_model(counts: ndarray, prior_lambda: float):
    """
    A model that assumes counts are generated by 2 different poisson processes:
    * counts up to switchpoint (not inclusive) ~ Poisson(early_rate)
    * counts from switchpoint on (inclusive) ~ Poisson(late_rate)

    Parameters
    ----------

    counts :
        1 - dimensional array of counts
    prior_lambda :
        parameter for exponential prior; 1 / prior_lambda is the mean of the exponential


    Returns
    -------

    pm.Model :
        the model instance

    Based on https://docs.pymc.io/notebooks/getting_started.html#Case-study-2:-Coal-mining-disasters
    """
    model = pm.Model()
    with model:
        idxs = np.arange(len(counts))
        lower_idx = idxs[1]
        upper_idx = idxs[-1]
        mid = (upper_idx - lower_idx) // 2

        switchpoint = pm.DiscreteUniform(
            "switchpoint", lower=lower_idx, upper=upper_idx, testval=mid
        )

        early_rate = pm.Exponential("early_rate", prior_lambda)
        late_rate = pm.Exponential("late_rate", prior_lambda)

        rate = pm.math.switch(switchpoint > idxs, early_rate, late_rate)

        pm.Poisson("counted", rate, observed=counts)
    return model


def make_2_poisson_model(counts: ndarray, labels: ndarray, prior_lambda: float):
    """
    counts where label == 0 ~ Poisson(mu1)
    counts where label == 1 ~ Poisson(mu2)

    TODO: Get rid of this and just run 2 separate Poisson models
    """
    assert counts.shape == labels.shape
    assert set(labels) == set([0, 1])

    counts1 = counts[labels == 0]
    counts2 = counts[labels == 1]

    model = pm.Model()

    with model:
        mu1 = pm.Exponential("mu1", prior_lambda)
        mu2 = pm.Exponential("mu2", prior_lambda)

        pm.Poisson("observed_counts1", mu1, observed=counts1)
        pm.Poisson("observed_counts2", mu2, observed=counts2)

        pm.Deterministic("k", mu2 / mu1)
    return model


def switchpoint_to_labels(size: int, switchpoint: int) -> ndarray:
    assert switchpoint > 0
    assert switchpoint <= size

    res = np.zeros(size, dtype=int)
    res[switchpoint:] = 1
    return res


class Test(unittest.TestCase):
    def test_switchpoint_model(self):
        counts = np.array([101, 99, 100, 1, 2, 1])
        model = make_switchpoint_model(counts, 1)
        with model:
            trace = pm.sample(10000)

        switchpoint = scipy.stats.mode(trace["switchpoint"])
        self.assertTrue(switchpoint, 3)

    def test_2poisson_model(self):
        counts = np.array([100, 100, 100, 100, 10, 10, 10, 10])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        model = make_2_poisson_model(counts, labels, 1 / 10)

        with model:
            trace = pm.sample(10000)
            mu1_mean = trace["mu1"].mean()
            mu2_mean = trace["mu2"].mean()
            self.assertTrue(abs(mu1_mean - 100) < 5)
            self.assertTrue(abs(mu2_mean - 10) < 5)
            print(
                "2poisson", trace["mu1"].mean(), trace["mu2"].mean(),
            )

    def test_switchpoint_to_labels(self):
        expected = np.array([0, 0, 1])
        res = switchpoint_to_labels(3, 2)

        self.assertTrue((expected == res).all())