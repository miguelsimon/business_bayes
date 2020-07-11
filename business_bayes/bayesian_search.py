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
    p_not_found = p_not_found_given_box * p[i] + p.sum() - p[i]

    p_new = p / p_not_found
    p_new[i] *= p_not_found_given_box
    return p_new


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
    search :
        search algorithm, follows the Search interface

    """

    def __init__(
        self, p0: ndarray, p_found_given_box: ndarray, costs: ndarray, search: Search
    ):

        self.p_found_given_box = p_found_given_box
        self.costs = costs

        self.search = search

        self.cost = 0
        self.found = False

        self.at_idx = np.random.choice(np.arange(p0.shape[0]), p=p0)

    def step(self) -> bool:
        if self.found:
            raise Exception

        i = self.search.search_next()

        if i == self.at_idx:
            p = self.p_found_given_box[i]
            self.found = np.random.random() < p

        if self.found:
            return True
        else:
            self.search.observe(i)
            return False

    def run(self) -> float:
        while self.step() == False:
            pass
        return self.cost


class Test(unittest.TestCase):
    def test_bayesian(self):
        p0 = np.array([0.5, 0.5])
        p_found_given_box = np.array([0.5, 0.5])
        costs = np.array([1, 1])

        search = BayesianSearch(p0, p_found_given_box, costs)

        sim = Simulate(p0, p_found_given_box, costs, search)

        sim.run()

    def test_random(self):
        p0 = np.array([0.5, 0.5])
        p_found_given_box = np.array([0.5, 0.5])
        costs = np.array([1, 1])

        search = RandomSearch(p0, p_found_given_box, costs)

        sim = Simulate(p0, p_found_given_box, costs, search)

        sim.run()
