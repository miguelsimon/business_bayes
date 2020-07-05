import unittest
from typing import Sequence

import numpy as np
import pymc3 as pm
from numpy import ndarray


def make_poisson_model():
    """
    This model fits a poisson mu to an array of counts
    representing eg. occurrences in a set of days.

    You'll need to construct a data dict to set the model data eg.

        model = make_poisson_model()

        data = {
            "lower_mu": 0,
            "upper_mu": 1000,
            "counts": np.array([100 for _ in range(20)]),
        }
        with model:
            pm.set_data(data)
            ...
    """
    model = pm.Model()

    with model:
        counts = pm.Data("counts", [])

        lower_mu = pm.Data("lower_mu", 0)
        upper_mu = pm.Data("upper_mu", 100)

        # prior for mu
        # mu = pm.Exponential('mu', 1)
        mu = pm.Uniform("mu", lower=lower_mu, upper=upper_mu)

        pm.Poisson("observed_counts", mu, observed=counts)
    return model


def get_count_percentiles(
    counts: ndarray,
    percentiles: Sequence[float],
    lower_mu: float,
    upper_mu: float,
    num_samples: int = 2000,
) -> ndarray:
    """
    For each count in an array of counts, this function:
    * fits a poisson model to the count
    * calculates the percentiles from the trace

    This is useful for eg. plotting a graph that gives a notion of confidence;
    we can plot the 25, 50, 75 percentile to easily do this

    Parameters
    ----------

    counts :
        1 - dimensional array of counts
    percentiles :
        sequence of percentiles to be calculated
    lower_mu :
        lower bound for uniform mu prior
    upper_mu :
        upper bound for uniform mu prior
    num_samples :
        samples to extract from the mcmc chain

    Returns
    -------

    ndarray
        (num_percentiles, num_count) - dimensional array
    """

    model = make_poisson_model()
    res = []

    for count in counts:
        data = {
            "lower_mu": lower_mu,
            "upper_mu": upper_mu,
            "counts": [count],
        }

        with model:
            pm.set_data(data)
            trace = pm.sample(num_samples, progressbar=False)
            mus = trace["mu"]
            out = np.percentile(mus, percentiles)
            res.append(out)
    arr = np.array(res)
    return arr.transpose()


class Test(unittest.TestCase):
    def test_poisson_model(self):
        model = make_poisson_model()
        data = {
            "lower_mu": 0,
            "upper_mu": 1000,
            "counts": np.array([100 for _ in range(20)]),
        }

        with model:
            pm.set_data(data)
            trace = pm.sample(5000)
            mus = trace["mu"]
            a, b, c = np.percentile(mus, [5, 50, 95])

            self.assertTrue(a < 100)
            self.assertTrue(c > 100)
            self.assertTrue(abs(b - 100) < 10)

    def test_get_count_percentiles(self):

        counts = np.array([5, 3, 4, 5])
        percentiles = [25, 50, 75]
        lower_mu = 0
        upper_mu = 100

        res = get_count_percentiles(counts, percentiles, lower_mu, upper_mu)
        self.assertEqual(res.shape, (3, 4))
