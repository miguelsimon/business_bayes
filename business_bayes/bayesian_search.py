import abc
import unittest
from typing import List

import numpy as np
from numpy import ndarray


def observe(p: ndarray, i: int, p_found_given_box: float) -> ndarray:
    """
    Get the posterior given the prior and the fact that box i has been searched
    and the object has not been found.

    Parameters
    ----------

    p :
        prior distribution of object location probabilities
    i :
        index of box that was searched
    p_found_given_box :
        probability for finding the object at box i after a search given that the
        object is indeed located at box i

    Returns
    -------

    ndarray :
        posterior probabilities
    """
    assert isinstance(p, ndarray)

    p_not_found_given_box = 1 - p_found_given_box
    p_not_found = p_not_found_given_box * p[i] + (1 - p[i])

    p_new = p / p_not_found
    p_new[i] *= p_not_found_given_box
    return p_new / p_new.sum()


def choose_next(p: ndarray, p_found_given_box: ndarray, costs: ndarray) -> int:
    """
    Apparently David Blackwell proved that the optimal search policy for
    this is choosing the index which maximizes:

        (p[i] * p_found_given_box[i]) / costs[i]


    Parameters
    ----------

    p :
        distribution of object location probabilities
    p_found_given_box :
        probability for finding the object at box i after a search given that the
        object is indeed located at box i
    costs :
        costs[i] is the cost of searching box[i]

    Returns
    -------

    int :
        index of box to be searched
    """
    assert isinstance(p, ndarray)
    assert isinstance(p_found_given_box, ndarray)

    to_maximize = p * p_found_given_box / costs
    return to_maximize.argmax()


class Search(metaclass=abc.ABCMeta):
    """
    Interface to specify a search algorithm
    """

    @abc.abstractmethod
    def search_next(self) -> int:
        """
        Return the index of the next box to be searched.
        """
        ...

    @abc.abstractmethod
    def observe(self, i: int):
        """
        Register the fact that box i has been searched and the object has not
        been found.
        """
        ...


class BayesianSearch(Search):
    """
    Implementation of the classic bayesian search algorithm.
    """

    ps: List[ndarray]
    p_found_given_box: ndarray
    costs: ndarray

    def __init__(self, p0: ndarray, p_found_given_box: ndarray, costs: ndarray):
        self.ps = [p0]
        self.p_found_given_box = p_found_given_box
        self.costs = costs

    def search_next(self) -> int:
        p = self.ps[-1]
        return choose_next(p, self.p_found_given_box, self.costs)

    def observe(self, i: int):
        p = self.ps[-1]
        p_found_given_box = self.p_found_given_box[i]
        p_new = observe(p, i, p_found_given_box)
        self.ps.append(p_new)

    def get_current_p(self) -> ndarray:
        return self.ps[-1]


class RandomSearch(Search):
    """
    Random search simply chooses a box index at random each time.
    This is useful as a baseline to compare against the bayesian search
    algorithm.
    """

    costs: ndarray

    def __init__(self, p0: ndarray, p_found_given_box: ndarray, costs: ndarray):
        self.costs = costs

    def search_next(self) -> int:
        return np.random.randint(0, self.costs.shape[0])

    def observe(self, i: int):
        pass


def perform_search(
    next_box_idx: int, location_idx: int, p_found_given_box: ndarray
) -> bool:
    """
    Stand-in for a real search

    Parameters
    ----------

    next_box_idx :
        index of box to be searched
    location_idx :
        index of the true location of object
    p_found_given_box :
        probability for finding the object at box i after a search given that the
        object is indeed located at box i

    Returns
    -------

    bool :
        True if the object was found, False otherwise
    """
    if next_box_idx == location_idx:
        prob = p_found_given_box[location_idx]
        found = np.random.binomial(1, prob)
        return found == 1
    else:
        return False


def simulate_bayes(
    p0: ndarray, p_found_given_box: ndarray, costs: ndarray, location_idx: int,
) -> float:
    """
    Runs a simulated bayes search step; probability of finding the object is sampled
    by using the p_found_given_box distribution.

    Parameters
    ----------

    p0 :
        initial probabilities for location
    p_found_given_box :
        probability for finding the object at box i after a search given that the
        object is indeed located at box i
    costs :
        cost of searching box i
    location_idx :
        true location of the object

    Returns
    -------

    float :
        cost
    """

    cost = 0
    p = p0

    while 1:
        next_box_idx = choose_next(p, p_found_given_box, costs)
        cost += costs[next_box_idx]
        found = perform_search(next_box_idx, location_idx, p_found_given_box)
        if found:
            break
        else:
            p = observe(p, next_box_idx, p_found_given_box[next_box_idx])

    return cost


def simulate_random(
    p_found_given_box: ndarray, costs: ndarray, location_idx: int,
) -> float:
    """
    Runs a simulated random search step; probability of finding the object is sampled
    by using the p_found_given_box distribution.

    Parameters
    ----------

    p_found_given_box :
        probability for finding the object at box i after a search given that the
        object is indeed located at box i
    costs :
        cost of searching box i
    location_idx :
        true location of the object

    Returns
    -------

    float :
        cost
    """

    cost = 0
    n = len(costs)

    while 1:
        next_box_idx = np.random.randint(0, n)
        cost += costs[next_box_idx]
        found = perform_search(next_box_idx, location_idx, p_found_given_box)
        if found:
            break

    return cost


class Test(unittest.TestCase):
    def test_means(self):
        n = 20
        location_idx = 19
        p0 = np.ones(n) / n
        p_found_given_box = np.ones(n) * 0.1
        costs = np.ones(n)

        bayes_costs = [
            simulate_bayes(p0, p_found_given_box, costs, location_idx)
            for _ in range(1000)
        ]
        random_costs = [
            simulate_random(p_found_given_box, costs, location_idx) for _ in range(1000)
        ]

        bayes_mean = np.mean(bayes_costs)
        random_mean = np.mean(random_costs)

        self.assertTrue(bayes_mean < random_mean)

    def test_means2(self):
        n = 3
        location_idx = 0
        p0 = np.ones(n) / n
        p_found_given_box = np.ones(n) * 0.1
        costs = np.ones(n)

        bayes_costs = [
            simulate_bayes(p0, p_found_given_box, costs, location_idx)
            for _ in range(1000)
        ]
        random_costs = [
            simulate_random(p_found_given_box, costs, location_idx) for _ in range(1000)
        ]

        bayes_mean = np.mean(bayes_costs)
        random_mean = np.mean(random_costs)

        self.assertTrue(bayes_mean < random_mean)
