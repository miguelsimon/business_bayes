import unittest
from typing import Callable, List

import numpy as np
from numpy import ndarray


def observe(p: ndarray, i: int, p_found_given_box: float) -> ndarray:
    assert isinstance(p, ndarray)

    p_not_found_given_box = 1 - p_found_given_box
    p_not_found = p_not_found_given_box * p[i] + p.sum() - p[i]

    p_new = p / p_not_found
    p_new[i] *= p_not_found_given_box
    return p_new


def choose_next(p: ndarray, p_found_given_box: ndarray, costs: ndarray) -> int:
    assert isinstance(p, ndarray)
    assert isinstance(p_found_given_box, ndarray)

    to_maximize = p * p_found_given_box / costs
    return to_maximize.argmax()


class BayesianSearch:
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


class RandomSearch:
    costs: ndarray

    def __init__(self, p0: ndarray, p_found_given_box: ndarray, costs: ndarray):
        self.costs = costs

    def search_next(self) -> int:
        return np.random.randint(0, self.costs.shape[0])

    def observe(self, i: int):
        pass


class Simulate:
    def __init__(
        self,
        p0: ndarray,
        p_found_given_box: ndarray,
        costs: ndarray,
        search_next: Callable[[], int],
        observe: Callable[[int], None],
    ):

        self.p_found_given_box = p_found_given_box
        self.costs = costs

        self.search_next = search_next
        self.observe = observe

        self.cost = 0
        self.found = False

        self.at_idx = np.random.choice(np.arange(p0.shape[0]), p=p0)

    def step(self) -> bool:
        if self.found:
            raise Exception

        i = self.search_next()

        if i == self.at_idx:
            p = self.p_found_given_box[i]
            self.found = np.random.random() < p

        if self.found:
            return True
        else:
            self.observe(i)
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

        sim = Simulate(p0, p_found_given_box, costs, search.search_next, search.observe)

        sim.run()

    def test_random(self):
        p0 = np.array([0.5, 0.5])
        p_found_given_box = np.array([0.5, 0.5])
        costs = np.array([1, 1])

        search = RandomSearch(p0, p_found_given_box, costs)

        sim = Simulate(p0, p_found_given_box, costs, search.search_next, search.observe)

        sim.run()
