import unittest
import pytest

from code import DistributionShift
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
import random


class TestDistributionShift(unittest.TestCase):
    def setUp(self):
        self.X = 2
        self.y = 3

    def test_strategy_assert(self):
        with pytest.raises(ValueError):
            DistributionShift(strategy='aa')
