import abc
import unittest
from typing import List

import numpy as np
from numpy import ndarray


def observe(p: ndarray, i: int, p_found_given_box: float) -> ndarray:
    """
    Get the posterior given the prior and the fact that box i has been searched
    and the object has not been found.
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
    Stand-in for performing a search: return True if found, False otherwise
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

    cost = 0
    n = len(costs)

    while 1:
        next_box_idx = np.random.randint(0, n)
        cost += costs[next_box_idx]
        found = perform_search(next_box_idx, location_idx, p_found_given_box)
        if found:
            break

    return cost


class Simulate:
    """
    Run a search simulation given a problem specification and a search algorithm.

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
    search :
        search algorithm, follows the Search interface

    """

    def __init__(
        self,
        p0: ndarray,
        p_found_given_box: ndarray,
        costs: ndarray,
        location_idx: int,
        search: Search,
    ):

        self.p_found_given_box = p_found_given_box
        self.costs = costs
        self.location_idx = location_idx
        self.search = search

        self.cost = 0
        self.found = False

    def step(self) -> bool:
        if self.found:
            raise Exception

        i = self.search.search_next()
        self.cost += self.costs[i]

        if i == self.location_idx:
            p = self.p_found_given_box[i]
            found = np.random.binomial(1, p)
            self.found = found == 1

        if self.found:
            return True
        else:
            self.search.observe(i)
            return False

    def run(self) -> float:
        while self.step() == False:
            pass
        return self.cost


def gather_data_points(p0, p_found_given_box, costs, location_idx, cls, num=10000):
    total_costs = []

    for i in range(num):
        np.random.seed(i)
        search = cls(p0, p_found_given_box, costs)
        sim = Simulate(p0, p_found_given_box, costs, location_idx, search)
        total_costs.append(sim.run())
    return total_costs


class Test(unittest.TestCase):
    def test_bayesian(self):
        p0 = np.array([0.5, 0.5])
        p_found_given_box = np.array([0.5, 0.5])
        costs = np.array([1, 1])

        search = BayesianSearch(p0, p_found_given_box, costs)

        sim = Simulate(p0, p_found_given_box, costs, 0, search)

        sim.run()

    def test_random(self):
        p0 = np.array([0.5, 0.5])
        p_found_given_box = np.array([0.5, 0.5])
        costs = np.array([1, 1])

        search = RandomSearch(p0, p_found_given_box, costs)

        sim = Simulate(p0, p_found_given_box, costs, 0, search)

        sim.run()

    def test_means(self):
        n = 20
        location_idx = 19
        p0 = np.ones(n) / n
        p_found_given_box = np.ones(n) * 0.1
        costs = np.ones(n)

        bayes_costs = gather_data_points(
            p0, p_found_given_box, costs, location_idx, BayesianSearch
        )
        random_costs = gather_data_points(
            p0, p_found_given_box, costs, location_idx, RandomSearch
        )

        bayes_mean = np.mean(bayes_costs)
        random_mean = np.mean(random_costs)

        self.assertTrue(bayes_mean < random_mean)

    def test_means2(self):
        n = 3
        location_idx = 0
        p0 = np.ones(n) / n
        p_found_given_box = np.ones(n) * 0.1
        costs = np.ones(n)

        bayes_costs = gather_data_points(
            p0, p_found_given_box, costs, location_idx, BayesianSearch
        )
        random_costs = gather_data_points(
            p0, p_found_given_box, costs, location_idx, RandomSearch
        )

        bayes_mean = np.mean(bayes_costs)
        random_mean = np.mean(random_costs)

        self.assertTrue(bayes_mean < random_mean)
