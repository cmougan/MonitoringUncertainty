import unittest

from code import *
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
import random
from doubt.datasets import PowerPlant


class TestDistributionShift(unittest.TestCase):
    def setUp(self):

        X, y = PowerPlant().split()
        self.X = X
        self.y = y

    def test_strategy_assert(self):
        db = DistributionShift()
        np.testing.assert_equal(2, 1 + 1)
